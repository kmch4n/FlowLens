"""Five-part live status strip."""

from dataclasses import dataclass

from PySide6.QtWidgets import QFrame, QHBoxLayout

from flowlens.ui.widgets import StatusIndicator


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    """Presentation snapshot for the bottom live status rail."""

    microphone_level: float
    loopback_level: float
    asr: str
    delay_ms: int
    analysis: str
    saved: str


class StatusStrip(QFrame):
    """Render microphone, PC audio, ASR, analysis, and save statuses separately."""

    def __init__(self) -> None:
        super().__init__()
        self.microphone_status = StatusIndicator("Microphone")
        self.pc_audio_status = StatusIndicator("PC audio")
        self.asr_status = StatusIndicator("ASR")
        self.analysis_status = StatusIndicator("Analysis")
        self.save_status = StatusIndicator("Latest save")
        self._build_layout()

    def render(self, snapshot: StatusSnapshot) -> None:  # type: ignore[override]
        """Render all five statuses without aggregating their messages."""

        self.microphone_status.set_status(
            "success",
            f"{round(max(0.0, min(1.0, snapshot.microphone_level)) * 100)}%",
        )
        self.pc_audio_status.set_status(
            "success",
            f"{round(max(0.0, min(1.0, snapshot.loopback_level)) * 100)}%",
        )
        asr_state = (
            "error" if snapshot.asr.lower() in {"delayed", "stopped"} else "success"
        )
        self.asr_status.set_status(
            asr_state, f"{snapshot.asr} · {snapshot.delay_ms} ms"
        )
        analysis_state = "error" if "paused" in snapshot.analysis.lower() else "success"
        self.analysis_status.set_status(analysis_state, snapshot.analysis)
        save_state = "default" if snapshot.saved == "Not saved yet" else "success"
        self.save_status.set_status(save_state, snapshot.saved)

    def _build_layout(self) -> None:
        self.setProperty("flowlensRole", "statusStrip")
        self.setMinimumHeight(64)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)
        for label in (
            self.microphone_status,
            self.pc_audio_status,
            self.asr_status,
            self.analysis_status,
            self.save_status,
        ):
            label.setMinimumWidth(120)
            layout.addWidget(label)
