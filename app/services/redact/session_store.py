from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.models.redact import RedactAuditResponse, RedactOptions
from app.pipeline.page_state import PageProcessState


@dataclass
class RedactSession:
    session_id: str
    file_bytes: bytes
    filename: str
    options: RedactOptions
    page_states: list[PageProcessState] = field(default_factory=list)
    last_pdf: bytes | None = None
    last_audit: RedactAuditResponse | None = None
    custom_terms: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, RedactSession] = {}

    def put(self, session: RedactSession) -> None:
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> RedactSession | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


session_store = SessionStore()
