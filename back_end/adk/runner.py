# Notice:
# Carter Saar. Copyright (C) 2025 State of Utah
# Licensed under Apache License, Version 2.0 (Apache v2). This program is distributed on an "AS IS" BASIS, WITHOUT ANY WARRANTY OR CONDITIONS OF ANY KIND, either express or implied. See the Apache License, Version 2.0 (Apache v2) for more details.

"""
Runner wiring and the synchronous bridge Streamlit consumes.

Both services are in-memory. Large payloads go to the artifact service rather
than session state, which ADK's docs recommend and which keeps them
session-scoped instead of on disk.
"""

from __future__ import annotations

import asyncio
import uuid

from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.genai import types

from back_end.adk.agent import (
    STATE_EVIDENCE_WARNINGS,
    STATE_QUOTA_WARNING,
    WORKFLOW_NAME,
    build_initial_state,
    build_root_agent,
)

APP_NAME = "risk_assessment_app"


class AssessmentRun:
    """Holds the services for one run so state and artifacts stay inspectable."""

    def __init__(self, session_service, artifact_service, session_id, user_id):
        self.session_service = session_service
        self.artifact_service = artifact_service
        self.session_id = session_id
        self.user_id = user_id

    async def state(self) -> dict:
        session = await self.session_service.get_session(
            app_name=APP_NAME, user_id=self.user_id, session_id=self.session_id
        )
        return dict(session.state) if session else {}

    async def artifact(self, filename: str) -> str | None:
        part = await self.artifact_service.load_artifact(
            app_name=APP_NAME, user_id=self.user_id, session_id=self.session_id, filename=filename
        )
        if part is None:
            return None
        blob = getattr(part, "inline_data", None)
        if blob is not None and blob.data is not None:
            return blob.data.decode("utf-8", errors="replace")
        return getattr(part, "text", None)

    async def artifact_names(self) -> list[str]:
        return await self.artifact_service.list_artifact_keys(
            app_name=APP_NAME, user_id=self.user_id, session_id=self.session_id
        )


async def run_assessment_async(
    *, api_key, model_id, temperature, matrix_df, inputs, client, on_progress=None
):
    """
    Execute one assessment.

    Returns (findings, evidence_warnings, quota_warning, AssessmentRun).

    on_progress, when supplied, is called with each user-facing progress message
    the graph emits. ADK separates Event.message (rendered to the user) from
    Event.output (passed to the next node), which is what makes genuine
    per-category progress reporting possible.
    """
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    session_id = f"run-{uuid.uuid4().hex[:12]}"
    user_id = "analyst"

    # Everything that varies per assessment is seeded here, so the graph itself
    # does not change shape with the matrix.
    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id,
        state=build_initial_state(matrix_df, inputs, model_id),
    )

    workflow = build_root_agent(
        api_key=api_key,
        model_id=model_id,
        temperature=temperature,
        inputs=inputs,
        client=client,
    )

    runner = Runner(
        app_name=APP_NAME,
        node=workflow,
        session_service=session_service,
        artifact_service=artifact_service,
    )

    # new_message must be a types.Content; a bare string raises AttributeError.
    message = types.Content(role="user", parts=[types.Part(text="Run the risk assessment.")])

    payload = None
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message
    ):
        if on_progress is not None:
            text = _progress_text(event)
            if text:
                on_progress(text)
        if getattr(event, "output", None):
            payload = event.output

    run = AssessmentRun(session_service, artifact_service, session_id, user_id)
    state = await run.state()
    if not isinstance(payload, dict):
        payload = {"findings": [], "warnings": state.get(STATE_EVIDENCE_WARNINGS, [])}
    return (
        payload.get("findings", []),
        payload.get("warnings", []),
        state.get(STATE_QUOTA_WARNING),
        run,
    )


def _progress_text(event) -> str | None:
    """
    Text of a progress event, or None if this event is not one.

    Only the orchestrator's own Event(message=...) counts. The per-category
    agents also emit their model output as content, and without this filter the
    raw assessment JSON was surfacing in the status line as if it were progress.
    """
    if getattr(event, "author", None) != WORKFLOW_NAME:
        return None
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            return text.strip()
    return None


def run_assessment(**kwargs):
    """
    Synchronous entry point for Streamlit.

    A fresh event loop per run, with a fresh model and runner built inside it:
    ADK's Gemini.api_client is a cached_property, so reusing a model across
    loops can leave it holding a transport bound to a closed loop.
    """
    return asyncio.run(run_assessment_async(**kwargs))
