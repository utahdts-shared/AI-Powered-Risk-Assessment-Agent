# Notice:
# Carter Saar. Copyright (C) 2025 State of Utah
# Licensed under Apache License, Version 2.0 (Apache v2). This program is distributed on an "AS IS" BASIS, WITHOUT ANY WARRANTY OR CONDITIONS OF ANY KIND, either express or implied. See the Apache License, Version 2.0 (Apache v2) for more details.

"""
Authentication and model discovery for the Google Gen AI SDK (Gemini).

This module is deliberately free of Streamlit imports: it returns values and
raises typed errors, and the UI layer decides how to display them.

Note for google-genai >= 2.x: hold the client in a variable. A temporary such as
``genai.Client(api_key=k).models.list()`` is garbage collected mid-call and
reports a closed transport instead of the real API error.
"""

from dataclasses import dataclass

from google import genai
from google.genai import errors

# Models this app is willing to run, in preference order. The account's own list
# is intersected with this one, so a retired model drops out of the picker
# without a redeploy and a newly published one never silently appears. That list
# is much broader -- image, audio and embedding models among them -- and is not
# a signal that a model will answer: some are listed but rejected on use, so
# this is limited to ones confirmed to work.
ALLOWED_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
)

DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Fallbacks used only when the API does not report a model's temperature range.
FALLBACK_DEFAULT_TEMPERATURE = 1.0
FALLBACK_MAX_TEMPERATURE = 2.0


@dataclass(frozen=True)
class ModelInfo:
    """A selectable model and the temperature range the API reports for it."""

    name: str
    default_temperature: float
    max_temperature: float

# Standard "AIza..." keys are rejected by the Gemini API as of September 2026;
# new AI Studio keys are issued in a different format.
# See https://ai.google.dev/gemini-api/docs/api-key
_KEY_TYPE_HELP = (
    "Create a new key at https://aistudio.google.com/apikey — keys are now issued "
    "in the supported format. Older standard keys (the shorter `AIza...` style) are "
    "no longer accepted by the Gemini API."
)


class ApiKeyError(Exception):
    """An API key could not be used. ``str(...)`` is safe to show to a user."""


def normalize_api_key(api_key):
    """
    The single definition of what this app considers the user's key.

    Every consumer uses this, so the key that gets validated is byte-for-byte
    the key later requests are made with. A pasted key routinely picks up a
    trailing space or newline.
    """
    return (api_key or "").strip()


def create_client(api_key):
    """
    Build a Gemini client. Does not perform any network call.

    Raises:
        ApiKeyError: if no key was supplied or the client cannot be constructed.
    """
    key = normalize_api_key(api_key)
    if not key:
        raise ApiKeyError("Enter a Gemini API key to continue.")
    try:
        return genai.Client(api_key=key)
    except Exception as exc:
        raise ApiKeyError(f"Could not initialize the Gemini client: {exc}") from exc


def list_available_models(client):
    """
    Validate the key and return the models this app can use, in preference order.

    This is a real authenticated round-trip, so it doubles as key validation.

    Each model reports its own default and maximum temperature, so the UI can
    offer a range appropriate to the selected model rather than one hard-coded
    value. Gemini 3.x models report a default of 1.0, which is also what Google
    recommends for them.

    Returns:
        list[ModelInfo]: allowlisted models available to this key, best first.

    Raises:
        ApiKeyError: with user-facing guidance if the key is rejected.
    """
    try:
        models = list(client.models.list())
    except Exception as exc:
        raise ApiKeyError(describe_api_error(exc)) from exc

    available = {}
    for model in models:
        actions = model.supported_actions or ()
        if "generateContent" not in actions:
            continue
        available[(model.name or "").removeprefix("models/")] = model

    usable = []
    for name in ALLOWED_MODELS:
        model = available.get(name)
        if model is None:
            continue
        usable.append(
            ModelInfo(
                name=name,
                default_temperature=float(
                    model.temperature
                    if model.temperature is not None
                    else FALLBACK_DEFAULT_TEMPERATURE
                ),
                max_temperature=float(
                    model.max_temperature
                    if model.max_temperature is not None
                    else FALLBACK_MAX_TEMPERATURE
                ),
            )
        )

    if not usable:
        raise ApiKeyError(
            "This key works, but none of the models this app supports are available to it. "
            f"Expected one of: {', '.join(ALLOWED_MODELS)}."
        )
    return usable


def describe_api_error(exc):
    """Translate an SDK exception into a message worth showing a user."""
    if isinstance(exc, errors.ClientError):
        code = getattr(exc, "code", None)
        text = str(exc)
        if code == 400 and "API_KEY_INVALID" in text:
            return f"API key not valid. {_KEY_TYPE_HELP}"
        if code in (401, 403):
            return (
                "Key rejected. Check that it was copied in full and that the Gemini API is "
                f"enabled for it. {_KEY_TYPE_HELP}"
            )
        if code == 429:
            return (
                "The key is valid but is being rate limited right now. Free-tier keys have low "
                "per-minute limits — wait a moment and try again."
            )
        return f"The Gemini API rejected the request ({code}): {text[:200]}"

    if isinstance(exc, errors.ServerError):
        return f"The Gemini API is having trouble right now ({getattr(exc, 'code', '5xx')}). Try again shortly."

    if isinstance(exc, RuntimeError) and "client has been closed" in str(exc):
        # Guard for the temporary-client GC trap described in the module docstring.
        return (
            "Internal error: the Gemini client was released before its request completed. "
            "This is a bug — please report it."
        )

    return f"Could not reach the Gemini API: {exc}"
