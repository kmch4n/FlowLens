"""Presentation paragraphs over immutable transcript records."""

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
)

from flowlens.domain.enums import AudioSource
from flowlens.domain.messages import TranscriptRecord
from flowlens.ui.transcript_model import TranscriptListModel

_ROOT_INDEX = QModelIndex()
_MAX_PARAGRAPH_CHARS = 180
_MAX_ME_PAUSE_MS = 1200
_SENTENCE_ENDINGS = ("。", "！", "？", ".", "!", "?")  # noqa: RUF001


class GroupedTranscriptModel(QAbstractListModel):
    """Combine short ME continuations while keeping source records untouched."""

    def __init__(self, source: TranscriptListModel) -> None:
        super().__init__(source)
        self._source = source
        self._groups = self._build_groups()
        source.rowsInserted.connect(self._source_rows_inserted)
        source.modelReset.connect(self._source_reset)

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX
    ) -> int:
        if parent.isValid():
            return 0
        return len(self._groups)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = int(Qt.ItemDataRole.DisplayRole),
    ) -> object:
        if not index.isValid() or not 0 <= index.row() < len(self._groups):
            return None
        group = self._groups[index.row()]
        source = group[0].source
        text = self._joined_text(group)
        if role == int(Qt.ItemDataRole.DisplayRole):
            return f"{self._source.source_label(source)} · {text}"
        if role == TranscriptListModel.source_role:
            return source
        if role == TranscriptListModel.text_role:
            return text
        if role == TranscriptListModel.record_role:
            return group[0]
        return None

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def _source_rows_inserted(self, *_: object) -> None:
        updated = self._build_groups()
        if updated[:-1] == self._groups and len(updated) == len(self._groups) + 1:
            position = len(self._groups)
            self.beginInsertRows(_ROOT_INDEX, position, position)
            self._groups = updated
            self.endInsertRows()
            return
        if (
            self._groups
            and len(updated) == len(self._groups)
            and updated[:-1] == self._groups[:-1]
            and updated[-1][:-1] == self._groups[-1]
        ):
            self._groups = updated
            last = self.index(len(updated) - 1, 0)
            self.dataChanged.emit(last, last)
            return
        self._replace_all(updated)

    def _source_reset(self) -> None:
        self._replace_all(self._build_groups())

    def _replace_all(self, groups: list[tuple[TranscriptRecord, ...]]) -> None:
        self.beginResetModel()
        self._groups = groups
        self.endResetModel()

    def _build_groups(self) -> list[tuple[TranscriptRecord, ...]]:
        groups: list[tuple[TranscriptRecord, ...]] = []
        for record in self._source.records():
            if groups and self._can_join(groups[-1], record):
                groups[-1] = (*groups[-1], record)
            else:
                groups.append((record,))
        return groups

    @staticmethod
    def _can_join(
        group: tuple[TranscriptRecord, ...], record: TranscriptRecord
    ) -> bool:
        previous = group[-1]
        pause_ms = record.session_start_ms - previous.session_end_ms
        return (
            previous.source is AudioSource.ME
            and record.source is AudioSource.ME
            and 0 <= pause_ms <= _MAX_ME_PAUSE_MS
            and not previous.text.rstrip().endswith(_SENTENCE_ENDINGS)
            and sum(len(item.text) for item in group) + len(record.text)
            <= _MAX_PARAGRAPH_CHARS
        )

    @staticmethod
    def _joined_text(group: tuple[TranscriptRecord, ...]) -> str:
        result = group[0].text
        for record in group[1:]:
            text = record.text
            separator = (
                " "
                if result[-1:].isascii()
                and result[-1:].isalnum()
                and text[:1].isascii()
                and text[:1].isalnum()
                else ""
            )
            result += separator + text
        return result
