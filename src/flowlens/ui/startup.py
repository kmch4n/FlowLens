"""Run local startup I/O outside the Qt GUI thread."""

from collections.abc import Callable

from PySide6.QtCore import QThread, Signal


class StartupTask(QThread):
    """Prepare plain Python application state without accessing any widget."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, prepare: Callable[[], object]) -> None:
        super().__init__()
        self._prepare = prepare

    def run(self) -> None:
        """Report preparation success or a privacy-safe failure category."""
        try:
            result = self._prepare()
        except Exception as error:
            self.failed.emit(type(error).__name__)
        else:
            self.completed.emit(result)
