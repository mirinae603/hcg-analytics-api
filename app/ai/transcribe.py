"""Voice input for the AI Analyst: a short recording -> text, through Azure OpenAI.

The chat stays a text chat. The browser records the question, POST /ai/transcribe sends the
clip to an Azure OpenAI speech-to-text deployment (gpt-4o-mini-transcribe by choice:
about $0.003 a minute of audio, so a ten-second question costs a fraction of a rupee), and the
text comes back into the chat box, where the user reads it and presses Send. Nothing new
answers questions: every spoken question goes through the same orchestrator, guards and
auditor as a typed one.

Nothing is kept. The audio is held in memory for the one Azure call and never written,
logged or cached; only the text goes back to the browser.

Configuration (environment, read on every call, so setting it needs only a restart of the
revision that reads it):
    AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT   the deployment name. Unset = voice input is off and
                                         GET /ai/status says so, so the UI hides the mic.
    AZURE_OPENAI_TRANSCRIBE_ENDPOINT     optional; default AZURE_OPENAI_ENDPOINT. Set both for
    AZURE_OPENAI_TRANSCRIBE_API_KEY      a separate resource, when the chat resource's region
                                         does not offer the transcribe model.
    AZURE_OPENAI_TRANSCRIBE_API_VERSION  optional; default TRANSCRIBE_API_VERSION.

Drug and hospital names are the words a general model mishears ("Keytruda", "Suchirayu"), so
each call carries a short hint list (Azure's `prompt`): the app's own terms plus the brands
with the most sales and the hospitals' distinctive names, read from the sales table.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)

# The transcribe models need an api-version from 2025-03-01-preview on.
TRANSCRIBE_API_VERSION = "2025-03-01-preview"
# A spoken question is seconds long; the browser stops at 60 s. 4 MB is far above a minute of
# the browser's Opus audio (~0.5 MB) and far below anything that would hurt the server.
MAX_AUDIO_BYTES = 4 * 1024 * 1024
# Below this there is no speech to send (a click on the mic and straight off again).
MIN_AUDIO_BYTES = 1024
# What the browsers record (MediaRecorder): Chrome/Edge/Firefox webm or ogg Opus, Safari mp4.
# The value is the file extension Azure reads the format from.
AUDIO_TYPES = {
    "audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4", "audio/x-m4a": "m4a",
    "audio/m4a": "m4a", "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav",
}
# Spoken-language choices: English (Indian English is read as English), Hindi, or let the
# model decide (a question that mixes the two).
LANGUAGES = {"en": "en", "hi": "hi", "auto": None}
# The app's own vocabulary, always in the hint.
BASE_TERMS = ("HCG", "formulary", "out of formulary", "non-formulary", "compliance", "margin",
              "MRP", "GRN", "cash and TPA", "scheme", "onco", "non-onco", "consumables", "OPD",
              "IPD", "SKU")
# How many brands and how long the hint may get: Azure reads only the first part of a long
# prompt, and the base terms must survive.
HINT_BRANDS = 40
HINT_MAX_CHARS = 900


class VoiceError(Exception):
    """A failure the user should read in plain words. `status` is the HTTP status to answer."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class Settings:
    deployment: str
    endpoint: str
    key: str
    api_version: str


def settings() -> Optional[Settings]:
    """The transcribe configuration, or None when voice input is not set up."""
    deployment = os.getenv("AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT", "").strip()
    key = (os.getenv("AZURE_OPENAI_TRANSCRIBE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
    if not deployment or not key:
        return None
    from app.ai import orchestrator
    endpoint = (os.getenv("AZURE_OPENAI_TRANSCRIBE_ENDPOINT") or orchestrator.AZURE_ENDPOINT).strip()
    version = (os.getenv("AZURE_OPENAI_TRANSCRIBE_API_VERSION") or TRANSCRIBE_API_VERSION).strip()
    return Settings(deployment, endpoint, key, version)


def configured() -> bool:
    return settings() is not None


@lru_cache(maxsize=2)
def _client(endpoint: str, key: str, version: str):
    # One client per configuration, as orchestrator._client_for does: a new one per clip would
    # pay a fresh TLS handshake every time. One retry: a clip is cheap to send again, and the
    # user is waiting on it.
    from openai import AzureOpenAI
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=version,
                       timeout=30.0, max_retries=1)


def media_type(content_type: Optional[str]) -> str:
    """'audio/webm;codecs=opus' -> 'audio/webm'; VoiceError 415 for anything that is not a
    recording the browsers make."""
    base = (content_type or "").split(";")[0].strip().lower()
    if base not in AUDIO_TYPES:
        raise VoiceError(415, "That recording format can't be read. Please try again from Chrome, Edge, Firefox or Safari.")
    return base


def check_size(n: int) -> None:
    if n > MAX_AUDIO_BYTES:
        raise VoiceError(413, "The recording is too long. Please keep a spoken question under a minute.")
    if n < MIN_AUDIO_BYTES:
        raise VoiceError(400, "Nothing was recorded. Hold the mic a little longer and speak your question.")


def _brand(desc) -> str:
    """The brand part of an item name: the words before the first one that holds a digit
    ('KEYTRUDA 100MG INJ VIAL' -> 'KEYTRUDA', 'ERBITUX 100MG' -> 'ERBITUX')."""
    words = []
    for w in str(desc or "").split():
        if re.search(r"\d", w):
            break
        words.append(w)
    return " ".join(words[:3]).strip(" .,-")


# Words every hospital name shares, which tell a listener nothing about which one is meant.
PLACE_STOPWORDS = {"HCG", "LLP", "PVT", "LTD", "HOSPITAL", "HOSPITALS", "CANCER", "CENTRE", "CENTER",
                   "INSTITUTE", "ONCOLOGY", "MULTISPECIALITY", "-"}
# Item categories whose brand names are spoken in questions (drugs, not gloves and syringes).
DRUG_CATEGORIES = ("Onco Drugs", "Other Drugs")


def _place(name) -> str:
    """A hospital's distinctive words ('HCG Suchirayu' -> 'Suchirayu', 'HCG KR, Bangalore' ->
    'KR Bangalore', 'HCG Aastha Cancer Centre , Ahmedabad' -> 'Aastha Ahmedabad')."""
    words = [w.strip(".-") for w in str(name or "").replace(",", " ").split()]
    return " ".join(w for w in words if w and w.upper() not in PLACE_STOPWORDS)


def hint_from(frame) -> str:
    """The hint for a sales_items frame (material_desc, mrp, hospital_name, category), or the
    base terms alone when there is none: the base terms, every hospital's distinctive name,
    then drug brands by MRP, biggest first, until the hint is full."""
    terms: list[str] = list(BASE_TERMS)
    if frame is not None and len(frame):
        terms += sorted({p for p in frame["hospital_name"].map(_place) if p})
        drugs = frame[frame["category"].isin(DRUG_CATEGORIES)]
        by_brand = (drugs.assign(brand=drugs["material_desc"].map(_brand))
                    .groupby("brand")["mrp"].sum().sort_values(ascending=False, kind="mergesort"))
        terms += [b for b in by_brand.index if len(b) > 2][:HINT_BRANDS]
    seen, out, size = set(), [], 0
    for t in terms:
        if t.upper() in seen:
            continue
        if size + len(t) + 2 > HINT_MAX_CHARS:
            break
        seen.add(t.upper())
        out.append(t)
        size += len(t) + 2
    return "Pharmacy analytics for HCG hospitals. Terms: " + ", ".join(out) + "."


@lru_cache(maxsize=2)
def _hint_for(version) -> str:
    from app.api import revenue as RV
    try:
        frame = RV.sales_items()[["material_desc", "mrp", "hospital_name", "category"]]
    except Exception:  # sales table not built: the base terms still help
        frame = None
    return hint_from(frame)


def vocabulary_hint() -> str:
    """The hint list for this build of the sales table (worked out again after a rebuild)."""
    from app.api import revenue as RV
    try:
        version = RV.version()
    except Exception:
        version = None
    return _hint_for(version)


def transcribe(audio: bytes, content_type: Optional[str], language: str = "en") -> str:
    """The text spoken in `audio`. VoiceError with a plain message on every failure."""
    cfg = settings()
    if cfg is None:
        raise VoiceError(503, "Voice input is not set up on this server yet.")
    if language not in LANGUAGES:
        raise VoiceError(400, "Unknown language. Use en, hi or auto.")
    mime = media_type(content_type)
    check_size(len(audio))
    kwargs = {"model": cfg.deployment, "file": (f"question.{AUDIO_TYPES[mime]}", audio, mime),
              "prompt": vocabulary_hint(), "response_format": "json"}
    if LANGUAGES[language]:
        kwargs["language"] = LANGUAGES[language]
    try:
        result = _client(cfg.endpoint, cfg.key, cfg.api_version).audio.transcriptions.create(**kwargs)
    except Exception as e:  # the audio and the key never reach the log, only what failed
        log.warning("voice transcription failed: %s", type(e).__name__)
        raise VoiceError(502, "Couldn't turn the recording into text. Please try again, or type the question.") from None
    text = (getattr(result, "text", None) or "").strip()
    if not text:
        raise VoiceError(422, "No words were heard in the recording. Please try again a little closer to the mic.")
    return text
