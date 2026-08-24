"""Source-to-mock mapping table helpers and optional NiceGUI panel.

Pure helpers stay importable without a NiceGUI page context. Integration
mounts ``build_mapping_panel`` and injects store callbacks. This module
does not log ``source_text`` or other PII values.

Only ``source_text`` (Name) and ``mock_value`` (Mock value) are shown or
editable here — matching applies to a name irrespective of whatever PII
category it was detected as, so there is nothing else to tag.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict


class MappingRow(TypedDict):
    """One verification-table row (ledger/dictionary fields only)."""

    source_text: str
    mock_value: str
    assignment_source: str
    hit_count: int
    mapping_id: str


class OverridePayload(TypedDict):
    """Validated edit sent to the injected ``on_override`` callback."""

    mapping_id: str
    mock_value: str


class CreatePayload(TypedDict):
    """Validated new-mapping payload sent to the injected ``on_create`` callback."""

    source_text: str
    mock_value: str


# Displayed in the table in this order; ``mapping_id`` is carried in row
# data (it's the ``row_key``) but not shown as its own column — the Edit/
# Delete actions column replaces the old copy-the-id-into-a-box workflow.
ROW_KEYS: tuple[str, ...] = (
    "source_text",
    "mock_value",
    "assignment_source",
    "hit_count",
    "mapping_id",
)

_DISPLAY_KEYS: tuple[str, ...] = ("source_text", "mock_value")

_COLUMN_LABELS: dict[str, str] = {
    "source_text": "Name",
    "mock_value": "Mock value",
    "assignment_source": "Assignment",
    "hit_count": "Hits",
    "mapping_id": "Mapping ID",
    "actions": "",
}


def _as_str(value: object) -> str:
    """Coerce a missing or None field to an empty string."""
    if value is None:
        return ""
    return str(value)


def _as_hit_count(value: object) -> int:
    """Coerce ``hit_count``; missing or None becomes 0."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    return 0


def rows_from_entries(entries: list[dict] | None) -> list[MappingRow]:
    """Normalize store/ledger dicts to table rows. Extra keys dropped.

    Args:
        entries: Ledger or mock-dictionary dicts, or None.

    Returns:
        Rows with only ``ROW_KEYS``, in input order. Non-dicts are skipped.
    """
    if not entries:
        return []

    rows: list[MappingRow] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "source_text": _as_str(item.get("source_text")),
                "mock_value": _as_str(item.get("mock_value")),
                "assignment_source": _as_str(item.get("assignment_source")),
                "hit_count": _as_hit_count(item.get("hit_count")),
                "mapping_id": _as_str(item.get("mapping_id")),
            }
        )
    return rows


def parse_override(mapping_id: str, mock_value: str) -> OverridePayload:
    """Validate an edit payload. Raises ValueError on bad input.

    Args:
        mapping_id: Dictionary/ledger mapping identifier.
        mock_value: Replacement mock text (must be non-blank after strip).

    Returns:
        Stripped ``mapping_id``/``mock_value``.

    Raises:
        ValueError: If ``mock_value``/``mapping_id`` is empty after strip.
    """
    mapping_id = mapping_id.strip()
    mock_value = mock_value.strip()
    if mock_value == "":
        raise ValueError("mock_value is required")
    if mapping_id == "":
        raise ValueError("mapping_id is required")
    return {"mapping_id": mapping_id, "mock_value": mock_value}


def parse_create(source_text: str, mock_value: str) -> CreatePayload:
    """Validate a brand-new mapping payload. Raises ValueError on bad input.

    Args:
        source_text: Original text to map (must be non-blank after strip).
        mock_value: Replacement mock text (must be non-blank after strip).

    Returns:
        Stripped fields ready for ``MockDictionaryStoreProtocol.upsert``.

    Raises:
        ValueError: If ``source_text``/``mock_value`` is empty after strip.
    """
    source_text = source_text.strip()
    mock_value = mock_value.strip()
    if source_text == "":
        raise ValueError("source_text is required")
    if mock_value == "":
        raise ValueError("mock_value is required")
    return {"source_text": source_text, "mock_value": mock_value}


def build_mapping_toolbar(
    on_download_mappings: Callable[[], None],
    on_download_template: Callable[[], None],
    on_import_file: Callable[[object], None],
) -> None:
    """NiceGUI: CSV download/upload controls for the mock dictionary.

    Download buttons trigger a browser save (integration builds the CSV via
    ``app.services.pii.mapping_csv`` and calls ``ui.download``). The upload
    control auto-uploads a dropped/selected ``.csv`` straight to the
    callback — no submit step, no user identity involved.

    Args:
        on_download_mappings: Triggered by the "Download mappings" button.
        on_download_template: Triggered by the "Download template" button.
        on_import_file: Receives the NiceGUI upload event for a chosen CSV.
    """
    from nicegui import ui  # type: ignore[import-not-found]

    with ui.row().classes("w-full items-center gap-3 flex-wrap"):
        ui.button(
            "Download mappings",
            icon="download",
            on_click=on_download_mappings,
        ).props("color=primary unelevated").tooltip(
            "Save every current name → mock mapping as CSV"
        )
        ui.button(
            "Download template",
            icon="description",
            on_click=on_download_template,
        ).props("color=secondary unelevated").tooltip(
            "Blank starter CSV — same columns, ready to fill in and upload"
        )
        with ui.element("div").classes("upload-dropzone compact min-w-[260px] grow"):
            ui.upload(
                label="Import mappings CSV",
                auto_upload=True,
                on_upload=on_import_file,
            ).props('accept=".csv" hide-upload-btn bordered color=primary').classes(
                "w-full"
            ).tooltip(
                "Add new mappings from a CSV — existing ones are never overwritten"
            )


def _actions_slot() -> str:
    """Quasar template: per-row Edit/Delete icon buttons.

    Buttons emit custom ``edit``/``delete`` events carrying the row dict —
    ``table.on('edit', ...)``/``table.on('delete', ...)`` below receive it
    as the event's ``args``.
    """
    return r"""
        <q-td :props="props" key="actions" class="text-right">
            <q-btn flat dense round size="sm" icon="edit" color="primary"
                   @click="() => $parent.$emit('edit', props.row)" />
            <q-btn flat dense round size="sm" icon="delete" color="negative"
                   @click="() => $parent.$emit('delete', props.row)" />
        </q-td>
    """


def build_mapping_panel(
    entries: list[dict],
    on_override: Callable[[OverridePayload], None],
    on_refresh: Callable[[], None],
    on_create: Callable[[CreatePayload], None] | None = None,
    on_delete: Callable[[str], None] | None = None,
) -> None:
    """NiceGUI: searchable table + Add/Edit/Delete dialogs for the dictionary.

    Every row has Edit and Delete actions that open a small dialog instead
    of the old "select a row, copy its id into a box below, click
    Override" flow — the id travels with the row data internally, so
    there's nothing to copy. An "Add mapping" button opens the same style
    of dialog for adding a name that was never auto-detected. Only the
    name and its mock value are ever shown or edited — the patch applies
    to that name irrespective of what PII category it's found under.

    Args:
        entries: Current ledger or dictionary dicts.
        on_override: Sync callback invoked with a validated edit payload.
        on_refresh: Sync callback for the Refresh button (no parse).
        on_create: Sync callback for a validated new-mapping payload. If
            omitted, the "Add mapping" button is not shown.
        on_delete: Sync callback receiving a ``mapping_id`` to delete. If
            omitted, per-row Delete buttons are not shown.
    """
    from nicegui import ui  # type: ignore[import-not-found]

    rows = rows_from_entries(entries)
    columns = [
        {
            "name": key,
            "label": _COLUMN_LABELS[key],
            "field": key,
            "sortable": True,
            "align": "left",
        }
        for key in _DISPLAY_KEYS
    ]
    if on_delete is not None:
        columns.append(
            {"name": "actions", "label": "", "field": "actions", "sortable": False, "align": "right"}
        )

    def _open_mapping_dialog(
        *,
        title: str,
        source_text: str,
        source_readonly: bool,
        mock_value: str,
        on_save: Callable[[str, str], None],
    ) -> None:
        with ui.dialog() as dialog, ui.card().classes(
            "bg-slate-800 text-slate-100 gap-3 min-w-[420px] max-w-[90vw]"
        ):
            ui.label(title).classes("text-lg font-semibold text-white")
            source_input = (
                ui.input(label="Name", value=source_text)
                .classes("w-full")
                .props('outlined dark aria-label="Name"')
            )
            source_input.set_enabled(not source_readonly)
            mock_input = (
                ui.input(label="Mock value", value=mock_value)
                .classes("w-full")
                .props('outlined dark aria-label="Mock value"')
            )

            def _save() -> None:
                try:
                    on_save(source_input.value or "", mock_input.value or "")
                except ValueError as exc:
                    ui.notify(str(exc), type="warning")
                    return
                dialog.close()

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Save", icon="save", on_click=_save).props("color=primary unelevated")
        dialog.open()

    def _open_edit_dialog(row: dict) -> None:
        def _save(_source: str, mock_value: str) -> None:
            payload = parse_override(row.get("mapping_id", ""), mock_value)
            on_override(payload)

        _open_mapping_dialog(
            title="Edit mapping",
            source_text=str(row.get("source_text", "")),
            source_readonly=True,
            mock_value=str(row.get("mock_value", "")),
            on_save=_save,
        )

    def _open_create_dialog() -> None:
        if on_create is None:
            return

        def _save(source_text: str, mock_value: str) -> None:
            payload = parse_create(source_text, mock_value)
            on_create(payload)

        _open_mapping_dialog(
            title="Add mapping",
            source_text="",
            source_readonly=False,
            mock_value="",
            on_save=_save,
        )

    def _confirm_delete(row: dict) -> None:
        if on_delete is None:
            return
        with ui.dialog() as dialog, ui.card().classes("bg-slate-800 text-slate-100 gap-3"):
            ui.label("Delete this mapping?").classes("text-base font-semibold text-white")
            ui.label(
                f"Mock value \"{row.get('mock_value', '')}\" will no longer be "
                "recognized on future runs — a fresh auto mapping may be "
                "created instead if the same name reappears."
            ).classes("text-sm text-slate-400 max-w-md")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def _do_delete() -> None:
                    dialog.close()
                    on_delete(row.get("mapping_id", ""))

                ui.button("Delete", icon="delete", on_click=_do_delete).props("color=negative unelevated")
        dialog.open()

    with ui.column().classes("w-full gap-3 text-slate-100"):
        with ui.row().classes("w-full items-center justify-between gap-3 flex-wrap"):
            ui.label("Name → Mock").classes(
                "text-sm font-semibold text-teal-300 uppercase tracking-wide"
            )
            with ui.row().classes("items-center gap-2"):
                search_input = (
                    ui.input(placeholder="Search name or mock value…")
                    .props("outlined dense dark clearable")
                    .classes("w-64")
                )
                if on_create is not None:
                    with ui.element("div").classes("primary-cta"):
                        ui.button("Add mapping", icon="add", on_click=_open_create_dialog).props(
                            "color=primary unelevated"
                        )

        table = ui.table(
            columns=columns,
            rows=rows,
            row_key="mapping_id",
            pagination={"rowsPerPage": 15},
        ).classes("w-full bg-slate-700 text-slate-100")
        table.bind_filter_from(search_input, "value")
        if on_delete is not None:
            table.add_slot("body-cell-actions", _actions_slot())
            table.on("edit", lambda e: _open_edit_dialog(e.args))
            table.on("delete", lambda e: _confirm_delete(e.args))

        ui.button(
            "Refresh",
            icon="refresh",
            on_click=on_refresh,
        ).props("color=accent outline")
