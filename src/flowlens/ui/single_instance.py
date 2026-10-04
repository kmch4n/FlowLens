"""Windows-local single-instance ownership with existing-window activation."""

from pathlib import Path

from PySide6.QtCore import QLockFile, QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket


class SingleInstance(QObject):
    """Hold the OS endpoint until shutdown; never unlink another owner's endpoint."""

    activation_requested = Signal()

    def __init__(self, name: str, lock_path: Path) -> None:
        super().__init__()
        self._name = name
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = QLockFile(str(lock_path))
        self._lock.setStaleLockTime(0)
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self._server.newConnection.connect(self._activate)

    def acquire(self) -> bool:
        """Return true only for the owner; otherwise notify the existing owner."""
        if self._server.isListening():
            return True
        if self._lock.tryLock(0):
            if self._server.listen(self._name):
                return True
            self._lock.unlock()
            raise RuntimeError("Cannot open the FlowLens activation endpoint")
        if self._lock.error() != QLockFile.LockError.LockFailedError:
            raise RuntimeError("Cannot lock the FlowLens application directory")
        socket = QLocalSocket()
        socket.connectToServer(self._name)
        socket.waitForConnected(1000)
        socket.abort()
        return False

    def close(self) -> None:
        """Release ownership after application work has stopped."""
        self._server.close()
        self._lock.unlock()

    def _activate(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            if socket is not None:
                socket.abort()
                socket.deleteLater()
        self.activation_requested.emit()
