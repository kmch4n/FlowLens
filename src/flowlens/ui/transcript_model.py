"""Qt list model for immutable committed transcript rows."""

from dataclasses import dataclass
from typing import ClassVar

from PySide6.QtCore import (
    QAbstractListModel,
    QByteArray,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
    Signal,
)

from flowlens.asr.types import PartialTranscript
from flowlens.domain.enums import AudioSource
from flowlens.domain.messages import TranscriptRecord

_ROOT_INDEX = QModelIndex()


class ImmutableTranscriptError(ValueError):
    """Raised when a committed transcript row would be edited or replaced."""


@dataclass(frozen=True, slots=True)
class TranscriptDisplayRow:
    """Presentation-ready transcript row with source shape metadata."""

    record: TranscriptRecord
    source_label: str
    text: str

    @property
    def source(self) -> AudioSource:
        """Return the source used by tests and view code."""

        return self.record.source

    @property
    def sequence(self) -> int:
        """Return the committed transcript sequence."""

        return self.record.sequence


class TranscriptListModel(QAbstractListModel):
    """Expose committed transcript rows while keeping partials ephemeral."""

    partials_changed = Signal()

    source_role: ClassVar[int] = int(Qt.ItemDataRole.UserRole) + 1
    text_role: ClassVar[int] = int(Qt.ItemDataRole.UserRole) + 2
    record_role: ClassVar[int] = int(Qt.ItemDataRole.UserRole) + 3

    _SOURCE_RANK: ClassVar[dict[AudioSource, int]] = {
        AudioSource.ME: 0,
        AudioSource.OTHERS: 1,
    }
    _SOURCE_LABELS: ClassVar[dict[AudioSource, str]] = {
        AudioSource.ME: "● ME",
        AudioSource.OTHERS: "■ OTHERS",
    }

    def __init__(self) -> None:
        super().__init__()
        self._records: list[TranscriptRecord] = []
        self._partials: dict[AudioSource, PartialTranscript] = {}
        self._by_segment_id: dict[str, TranscriptRecord] = {}
        self._by_sequence: dict[int, TranscriptRecord] = {}

    def rowCount(
        self,
        parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX,
    ) -> int:
        """Return the number of committed rows."""

        if parent is not None and parent.isValid():
            return 0
        return len(self._records)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = int(Qt.ItemDataRole.DisplayRole),
    ) -> object:
        """Return display and structured roles for one committed row."""

        if not index.isValid() or not 0 <= index.row() < len(self._records):
            return None
        record = self._records[index.row()]
        if role == int(Qt.ItemDataRole.DisplayRole):
            return f"{self.source_label(record.source)} · {record.text}"
        if role == self.source_role:
            return record.source
        if role == self.text_role:
            return record.text
        if role == self.record_role:
            return record
        return None

    def roleNames(self) -> dict[int, QByteArray]:
        """Return stable role names for Qt consumers."""

        return {
            self.source_role: QByteArray(b"source"),
            self.text_role: QByteArray(b"text"),
            self.record_role: QByteArray(b"record"),
        }

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        """Expose rows as selectable but not editable."""

        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def commit(self, record: TranscriptRecord) -> None:
        """Add one immutable committed row in transcript chronological order."""

        if not isinstance(record, TranscriptRecord):
            raise TypeError("record must be a TranscriptRecord")
        self._reject_replacement(record)
        self.beginResetModel()
        self._records.append(record)
        self._records.sort(key=self._sort_key)
        self._by_segment_id[record.segment_id] = record
        self._by_sequence[record.sequence] = record
        self._partials.pop(record.source, None)
        self.endResetModel()

    def set_partial(self, source: AudioSource, partial: PartialTranscript) -> None:
        """Set or replace the visible ephemeral partial for one source."""

        if not isinstance(source, AudioSource):
            raise TypeError("source must be an AudioSource")
        if not isinstance(partial, PartialTranscript):
            raise TypeError("partial must be a PartialTranscript")
        if partial.source is not source:
            raise ValueError("partial source must match the source slot")
        if not partial.text:
            self.clear_partial(source)
            return
        self._partials[source] = partial
        self.partials_changed.emit()

    def clear_partial(self, source: AudioSource) -> None:
        """Clear one source's ephemeral partial."""

        if not isinstance(source, AudioSource):
            raise TypeError("source must be an AudioSource")
        if source in self._partials:
            self._partials.pop(source)
            self.partials_changed.emit()

    def partial(self, source: AudioSource) -> PartialTranscript | None:
        """Return the current ephemeral partial for one source."""

        if not isinstance(source, AudioSource):
            raise TypeError("source must be an AudioSource")
        return self._partials.get(source)

    def row(self, index: int) -> TranscriptDisplayRow:
        """Return one presentation row by committed index."""

        if type(index) is not int or not 0 <= index < len(self._records):
            raise IndexError("transcript row index out of range")
        record = self._records[index]
        return TranscriptDisplayRow(
            record, self.source_label(record.source), record.text
        )

    def records(self) -> tuple[TranscriptRecord, ...]:
        """Return committed rows in display order."""

        return tuple(self._records)

    @classmethod
    def source_label(cls, source: AudioSource) -> str:
        """Return the text-and-shape label for one source."""

        return cls._SOURCE_LABELS[source]

    def _reject_replacement(self, record: TranscriptRecord) -> None:
        existing_by_segment = self._by_segment_id.get(record.segment_id)
        existing_by_sequence = self._by_sequence.get(record.sequence)
        if existing_by_segment is not None or existing_by_sequence is not None:
            if existing_by_segment == record or existing_by_sequence == record:
                raise ImmutableTranscriptError("committed transcript already exists")
            raise ImmutableTranscriptError("committed transcript cannot be replaced")

    @classmethod
    def _sort_key(cls, record: TranscriptRecord) -> tuple[int, int, int]:
        return (
            record.session_start_ms,
            cls._SOURCE_RANK[record.source],
            record.sequence,
        )
