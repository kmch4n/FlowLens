"""Two-source ASR scheduling, partials, and chronological commits."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from flowlens.asr.commit import (
    ChronologicalCommitBuffer,
    StablePrefixTracker,
    choose_split_ms,
    is_transcript_content,
)
from flowlens.asr.ports import DecoderPort, SpeechDetectorPort
from flowlens.asr.types import (
    AsrWorkerConfig,
    CommitCandidate,
    DecodedToken,
    DecodeHypothesis,
    PartialTranscript,
)
from flowlens.asr.vad import UtteranceBoundaryTracker
from flowlens.audio.types import FRAME_DURATION_MS, AudioFrame
from flowlens.domain._validation import (
    ContractValidationError,
    require_non_negative_int,
)
from flowlens.domain.enums import AudioSource
from flowlens.domain.ids import new_ulid
from flowlens.domain.messages import TranscriptRecord

_SOURCE_RANK = {AudioSource.ME: 0, AudioSource.OTHERS: 1}


def _utc_now() -> datetime:
    value = datetime.now(UTC)
    return value.replace(microsecond=(value.microsecond // 1_000) * 1_000)


@dataclass(frozen=True, slots=True)
class AsrBatch:
    """One immutable set of ephemeral and committed ASR outputs."""

    partials: tuple[PartialTranscript, ...]
    committed: tuple[TranscriptRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.partials, tuple) or not all(
            isinstance(item, PartialTranscript) for item in self.partials
        ):
            raise ContractValidationError(
                "partials must be a tuple of PartialTranscript values"
            )
        if not isinstance(self.committed, tuple) or not all(
            isinstance(item, TranscriptRecord) for item in self.committed
        ):
            raise ContractValidationError(
                "committed must be a tuple of TranscriptRecord values"
            )


@dataclass(slots=True)
class _SourceState:
    boundary: UtteranceBoundaryTracker
    stable_prefix: StablePrefixTracker
    pending: deque[AudioFrame] = field(default_factory=deque)
    utterance: list[tuple[AudioFrame, bool]] = field(default_factory=list)
    last_decode_monotonic_ms: int | None = None
    last_partial_text: str = ""
    committed_through_ms: int = 0
    decoded_frame_count: int = 0
    deferred_tokens: tuple[DecodedToken, ...] = ()
    next_source_sample: int | None = None
    last_session_start_ms: int | None = None
    last_captured_monotonic_ms: int | None = None


class AsrEngine:
    """Schedule two independent stream states through one shared decoder."""

    def __init__(
        self,
        config: AsrWorkerConfig,
        decoder: DecoderPort,
        speech_detector: SpeechDetectorPort,
        segment_id_factory: Callable[[], str] = new_ulid,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(config, AsrWorkerConfig):
            raise ContractValidationError("config must be an AsrWorkerConfig")
        if not callable(getattr(decoder, "decode", None)):
            raise ContractValidationError("decoder must implement decode")
        if not callable(getattr(speech_detector, "is_speech", None)):
            raise ContractValidationError("speech_detector must implement is_speech")
        if not callable(segment_id_factory):
            raise ContractValidationError("segment_id_factory must be callable")
        if now is not None and not callable(now):
            raise ContractValidationError("now must be callable")
        self._config = config
        self._decoder = decoder
        self._speech_detector = speech_detector
        self._now = _utc_now if now is None else now
        self._commits = ChronologicalCommitBuffer(segment_id_factory, self._now)
        self._states = {
            source: self._new_state() for source in (AudioSource.ME, AudioSource.OTHERS)
        }
        self._last_process_monotonic_ms: int | None = None
        self._finalized = False

    def accept(self, frame: AudioFrame) -> None:
        """Validate and enqueue one canonical source frame."""

        if self._finalized:
            raise ContractValidationError("cannot accept input after engine finalized")
        if not isinstance(frame, AudioFrame):
            raise ContractValidationError("frame must be an AudioFrame")
        state = self._states[frame.source]
        expected_source_sample = (
            0 if state.next_source_sample is None else state.next_source_sample
        )
        if frame.source_start_sample != expected_source_sample:
            raise ContractValidationError(
                "source sample offsets must be contiguous and non-overlapping"
            )
        if state.last_session_start_ms is not None and (
            frame.session_start_ms < state.last_session_start_ms + FRAME_DURATION_MS
        ):
            raise ContractValidationError(
                "source session timestamps must be non-overlapping"
            )
        if state.last_captured_monotonic_ms is not None and (
            frame.captured_monotonic_ms
            < state.last_captured_monotonic_ms + FRAME_DURATION_MS
        ):
            raise ContractValidationError(
                "source capture timestamps must be non-overlapping"
            )
        state.next_source_sample = frame.source_end_sample
        state.last_session_start_ms = frame.session_start_ms
        state.last_captured_monotonic_ms = frame.captured_monotonic_ms
        state.pending.append(frame)

    def process_ready(self, now_monotonic_ms: int) -> AsrBatch:
        """Process all queued frames and every source whose cadence is due."""

        now_ms = self._observe_process_time(now_monotonic_ms)
        if self._finalized:
            return AsrBatch((), ())
        partials: list[PartialTranscript] = []
        committed: list[TranscriptRecord] = []
        self._drain_pending(now_ms, partials, committed)

        ready_sources = sorted(
            (
                source
                for source, state in self._states.items()
                if self._periodic_decode_due(state, now_ms)
            ),
            key=self._active_source_key,
        )
        for source in ready_sources:
            self._decode(source, now_ms, final=False, partials=partials)
            self._release_ready(committed)
        return AsrBatch(tuple(partials), tuple(committed))

    def finalize(self, now_monotonic_ms: int) -> AsrBatch:
        """Final-decode both source states and release all commits exactly once."""

        now_ms = self._observe_process_time(now_monotonic_ms)
        if self._finalized:
            return AsrBatch((), ())
        partials: list[PartialTranscript] = []
        committed: list[TranscriptRecord] = []
        self._drain_pending(now_ms, partials, committed)
        active_sources = sorted(
            (source for source, state in self._states.items() if state.utterance),
            key=self._active_source_key,
        )
        for source in active_sources:
            self._final_decode_and_reset(source, now_ms, partials)
        committed.extend(self._commits.finalize())
        self._finalized = True
        return AsrBatch(tuple(partials), tuple(committed))

    def backlog_ms(self, now_monotonic_ms: int) -> int:
        """Return the age of the globally oldest unprocessed audio frame."""

        now_ms = require_non_negative_int(now_monotonic_ms, "now_monotonic_ms")
        if self._finalized:
            return 0
        pending_times = [
            state.pending[0].captured_monotonic_ms
            for state in self._states.values()
            if state.pending
        ]
        if not pending_times:
            return 0
        return max(0, now_ms - min(pending_times))

    def _new_state(self) -> _SourceState:
        return _SourceState(
            boundary=UtteranceBoundaryTracker(
                self._config.silence_end_ms,
                self._config.max_utterance_ms,
            ),
            stable_prefix=StablePrefixTracker(self._config.stable_age_ms),
        )

    def _observe_process_time(self, value: int) -> int:
        now_ms = require_non_negative_int(value, "now_monotonic_ms")
        if (
            self._last_process_monotonic_ms is not None
            and now_ms < self._last_process_monotonic_ms
        ):
            raise ContractValidationError("now_monotonic_ms must not move backwards")
        self._last_process_monotonic_ms = now_ms
        return now_ms

    def _drain_pending(
        self,
        now_ms: int,
        partials: list[PartialTranscript],
        committed: list[TranscriptRecord],
    ) -> None:
        while True:
            pending_sources = [
                source for source, state in self._states.items() if state.pending
            ]
            if not pending_sources:
                return
            source = min(
                pending_sources,
                key=lambda item: (
                    self._states[item].pending[0].captured_monotonic_ms,
                    _SOURCE_RANK[item],
                ),
            )
            state = self._states[source]
            frame = state.pending.popleft()
            speech = self._speech_detector.is_speech(frame)
            boundary = state.boundary.observe(speech)
            if boundary == "INACTIVE":
                continue
            if not state.utterance:
                self._commits.set_frontier(source, frame.session_start_ms)
            state.utterance.append((frame, speech))
            if boundary == "END":
                self._final_decode_and_reset(source, now_ms, partials)
                self._release_ready(committed)
            elif boundary == "HARD_SPLIT":
                self._split_and_reset(source, now_ms, partials)
                self._release_ready(committed)

    def _periodic_decode_due(self, state: _SourceState, now_ms: int) -> bool:
        if not state.utterance:
            return False
        cadence_start = (
            state.utterance[0][0].captured_monotonic_ms
            if state.last_decode_monotonic_ms is None
            else state.last_decode_monotonic_ms
        )
        return now_ms - cadence_start >= self._config.partial_interval_ms

    def _active_source_key(self, source: AudioSource) -> tuple[int, int]:
        state = self._states[source]
        if state.decoded_frame_count < len(state.utterance):
            frontier_ms = state.utterance[state.decoded_frame_count][
                0
            ].captured_monotonic_ms
        else:
            frontier_ms = (
                state.utterance[-1][0].captured_monotonic_ms + FRAME_DURATION_MS
            )
        return (frontier_ms, _SOURCE_RANK[source])

    def _decode(
        self,
        source: AudioSource,
        now_ms: int,
        *,
        final: bool,
        partials: list[PartialTranscript],
        hypothesis: DecodeHypothesis | None = None,
        boundary_ms: int | None = None,
    ) -> DecodeHypothesis:
        state = self._states[source]
        if not state.utterance:
            return DecodeHypothesis(())
        decoded = (
            self._decoder.decode(
                b"".join(item[0].pcm_s16le for item in state.utterance)
            )
            if hypothesis is None
            else hypothesis
        )
        state.last_decode_monotonic_ms = now_ms
        state.decoded_frame_count = len(state.utterance)
        committed_tokens = state.stable_prefix.observe(decoded, now_ms, final)
        pending_tokens = state.deferred_tokens + committed_tokens
        if self._push_tokens(source, pending_tokens, boundary_ms):
            state.deferred_tokens = ()
        else:
            state.deferred_tokens = pending_tokens
        normalized = decoded.text
        if not final and is_transcript_content(normalized):
            self._emit_changed_partial(source, normalized, partials)
        return decoded

    def _push_tokens(
        self,
        source: AudioSource,
        tokens: tuple[DecodedToken, ...],
        boundary_ms: int | None,
    ) -> bool:
        text = "".join(token.text for token in tokens).strip()
        if not tokens or not is_transcript_content(text):
            return True
        state = self._states[source]
        available_ms = len(state.utterance) * FRAME_DURATION_MS
        commit_limit_ms = (
            available_ms if boundary_ms is None else min(available_ms, boundary_ms)
        )
        if (
            state.committed_through_ms >= commit_limit_ms
            or tokens[0].start_ms >= commit_limit_ms
        ):
            return False
        token_start = min(
            commit_limit_ms - FRAME_DURATION_MS,
            max(0, tokens[0].start_ms // FRAME_DURATION_MS * FRAME_DURATION_MS),
        )
        relative_start = max(state.committed_through_ms, token_start)
        relative_end = min(
            commit_limit_ms,
            (tokens[-1].end_ms + FRAME_DURATION_MS - 1)
            // FRAME_DURATION_MS
            * FRAME_DURATION_MS,
        )
        relative_end = max(relative_start + FRAME_DURATION_MS, relative_end)
        start_index = relative_start // FRAME_DURATION_MS
        end_index = relative_end // FRAME_DURATION_MS - 1
        start_frame = state.utterance[start_index][0]
        end_frame = state.utterance[end_index][0]
        source_start = start_frame.source_start_sample
        source_end = end_frame.source_end_sample
        session_start = start_frame.session_start_ms
        session_end = end_frame.session_start_ms + FRAME_DURATION_MS
        self._commits.push(
            CommitCandidate(
                source=source,
                text=text,
                session_start_ms=session_start,
                session_end_ms=session_end,
                source_start_sample=source_start,
                source_end_sample=source_end,
                committed_at=self._now(),
            )
        )
        state.committed_through_ms = relative_end
        self._commits.set_frontier(source, session_end)
        return True

    def _emit_changed_partial(
        self,
        source: AudioSource,
        text: str,
        partials: list[PartialTranscript],
    ) -> None:
        state = self._states[source]
        if text == state.last_partial_text:
            return
        first = state.utterance[0][0]
        last = state.utterance[-1][0]
        partials.append(
            PartialTranscript(
                source=source,
                text=text,
                session_start_ms=first.session_start_ms,
                session_end_ms=last.session_start_ms + FRAME_DURATION_MS,
                source_start_sample=first.source_start_sample,
                source_end_sample=last.source_end_sample,
            )
        )
        state.last_partial_text = text

    def _clear_partial(
        self,
        source: AudioSource,
        partials: list[PartialTranscript],
    ) -> None:
        state = self._states[source]
        if not state.last_partial_text or not state.utterance:
            return
        first = state.utterance[0][0]
        last = state.utterance[-1][0]
        partials.append(
            PartialTranscript(
                source=source,
                text="",
                session_start_ms=first.session_start_ms,
                session_end_ms=last.session_start_ms + FRAME_DURATION_MS,
                source_start_sample=first.source_start_sample,
                source_end_sample=last.source_end_sample,
            )
        )
        state.last_partial_text = ""

    def _final_decode_and_reset(
        self,
        source: AudioSource,
        now_ms: int,
        partials: list[PartialTranscript],
    ) -> None:
        state = self._states[source]
        if not state.utterance:
            return
        self._decode(source, now_ms, final=True, partials=partials)
        self._clear_partial(source, partials)
        self._commits.set_frontier(source, None)
        self._reset_utterance(state)

    def _split_and_reset(
        self,
        source: AudioSource,
        now_ms: int,
        partials: list[PartialTranscript],
    ) -> None:
        state = self._states[source]
        decoded = self._decoder.decode(
            b"".join(item[0].pcm_s16le for item in state.utterance)
        )
        available_ms = len(state.utterance) * FRAME_DURATION_MS
        chosen_split_ms = min(
            choose_split_ms(decoded),
            available_ms,
        )
        committed_floor_ms = (
            (state.committed_through_ms + FRAME_DURATION_MS - 1)
            // FRAME_DURATION_MS
            * FRAME_DURATION_MS
        )
        split_ms = min(
            available_ms,
            max(chosen_split_ms, committed_floor_ms),
        )
        prefix = DecodeHypothesis(
            tuple(token for token in decoded.tokens if token.start_ms < split_ms)
        )
        state.deferred_tokens = tuple(
            token for token in state.deferred_tokens if token.start_ms < split_ms
        )
        self._decode(
            source,
            now_ms,
            final=True,
            partials=partials,
            hypothesis=prefix,
            boundary_ms=split_ms,
        )
        self._clear_partial(source, partials)
        split_frames = split_ms // FRAME_DURATION_MS
        tail = state.utterance[split_frames:]
        self._reset_utterance(state)
        state.utterance.extend(tail)
        for _frame, speech in tail:
            state.boundary.observe(speech)
        if tail:
            self._commits.set_frontier(source, tail[0][0].session_start_ms)
            state.last_decode_monotonic_ms = now_ms
        else:
            self._commits.set_frontier(source, None)

    def _reset_utterance(self, state: _SourceState) -> None:
        state.utterance.clear()
        state.boundary = UtteranceBoundaryTracker(
            self._config.silence_end_ms,
            self._config.max_utterance_ms,
        )
        state.stable_prefix = StablePrefixTracker(self._config.stable_age_ms)
        state.last_decode_monotonic_ms = None
        state.last_partial_text = ""
        state.committed_through_ms = 0
        state.decoded_frame_count = 0
        state.deferred_tokens = ()

    def _release_ready(self, committed: list[TranscriptRecord]) -> None:
        committed.extend(self._commits.release_ready())
