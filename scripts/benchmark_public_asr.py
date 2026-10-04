"""Small public-data decoder experiment; never capture or play audio."""

import io
import json
import os
import time
import unicodedata
import urllib.request
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from flowlens.asr.kotoba_whisper import _reliable_segment


def normalize(text: str) -> str:
    """Normalize width and remove punctuation/spacing for Japanese CER."""
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", text)
        if unicodedata.category(char)[0] not in {"P", "Z", "C"}
    )


def distance(reference: str, hypothesis: str) -> int:
    """Calculate character edit distance using one dynamic-programming row."""
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    return previous[-1]


def main() -> None:
    """Compare fixed beam widths on 48 public clips, without private inputs."""
    destination = Path("build/reports/asr-public")
    destination.mkdir(parents=True, exist_ok=True)
    url = (
        "https://datasets-server.huggingface.co/rows?"
        "dataset=japanese-asr/ja_asr.reazonspeech_test&config=default"
        "&split=test&offset=0&length=48"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        rows = json.load(response)["rows"]
    samples = []
    for entry in rows:
        row = entry["row"]
        with urllib.request.urlopen(row["audio"][0]["src"], timeout=30) as response:
            audio = decode_audio(io.BytesIO(response.read()), sampling_rate=16000)
        samples.append((entry["row_idx"], normalize(row["transcription"]), audio))
    model_path = (
        Path(os.environ["LOCALAPPDATA"]) / "FlowLens/models/kotoba-whisper-v2.0-faster"
    )
    model = WhisperModel(
        str(model_path), device="cuda", compute_type="float16", local_files_only=True
    )
    measurements = []
    for index, reference, audio in samples:
        for beam in (1, 3, 5):
            started = time.perf_counter()
            segments, _ = model.transcribe(
                audio,
                language="ja",
                task="transcribe",
                beam_size=beam,
                temperature=0,
                condition_on_previous_text=False,
                word_timestamps=True,
                vad_filter=False,
            )
            decoded = list(segments)
            hypothesis = normalize("".join(segment.text for segment in decoded))
            filtered = normalize(
                "".join(
                    segment.text for segment in decoded if _reliable_segment(segment)
                )
            )
            measurements.append(
                {
                    "sample": index,
                    "beam": beam,
                    "reference_chars": len(reference),
                    "edits": distance(reference, hypothesis),
                    "filtered_edits": distance(reference, filtered),
                    "seconds": time.perf_counter() - started,
                    "audio_seconds": len(audio) / 16000,
                }
            )
        print(f"Completed public clip {index + 1}/48", flush=True)
    report = {
        "dataset": "japanese-asr/ja_asr.reazonspeech_test",
        "rows": [0, 47],
        "limitations": "Small fixed sample; not real-time two-source acceptance.",
        "measurements": measurements,
    }
    (destination / "beam-comparison.json").write_text(
        json.dumps(report, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
