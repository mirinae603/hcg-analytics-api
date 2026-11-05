# tests/test_transcribe.py — voice input for the AI Analyst (app/ai/transcribe.py, POST
# /ai/transcribe). Azure is replaced by a fake client that records what it was sent, so the
# tests pin the contract without a key: what is sent to Azure, what comes back to the chat,
# and every refusal in plain words.
from __future__ import annotations

import logging

import pandas as pd
import pytest

from app.ai import transcribe as V

CLIP = b"\x1aE\xdf\xa3" + b"\x00" * 4000          # a webm header and some bytes: 4 KB
WEBM = {"Content-Type": "audio/webm;codecs=opus"}


class FakeAzure:
    def __init__(self, text="What was formulary compliance at KR in August?", fail=False):
        self.text, self.fail, self.calls = text, fail, []
        self.audio = self
        self.transcriptions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("azure said no")
        return type("R", (), {"text": self.text})()


@pytest.fixture()
def voice(monkeypatch):
    """Voice input set up with a fake Azure; yields the fake to inspect."""
    monkeypatch.setenv("AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT", "gpt-4o-mini-transcribe")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AZURE_OPENAI_TRANSCRIBE_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_TRANSCRIBE_API_KEY", raising=False)
    fake = FakeAzure()
    monkeypatch.setattr(V, "_client", lambda endpoint, key, version: fake)
    monkeypatch.setattr(V, "vocabulary_hint", lambda: "Pharmacy analytics for HCG hospitals. Terms: HCG, KEYTRUDA.")
    return fake


@pytest.fixture()
def no_voice(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT", raising=False)


# ---------- status and login ----------
def test_status_says_whether_voice_is_set_up(client, no_voice, monkeypatch):
    assert client.get("/ai/status").json()["voice"] is False
    monkeypatch.setenv("AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT", "gpt-4o-mini-transcribe")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    assert client.get("/ai/status").json()["voice"] is True
    monkeypatch.delenv("AZURE_OPENAI_API_KEY")            # a deployment without a key is not set up
    assert client.get("/ai/status").json()["voice"] is False


def test_transcribe_needs_a_login(anon_client, voice):
    assert anon_client.post("/ai/transcribe", content=CLIP, headers=WEBM).status_code == 401
    assert voice.calls == []


def test_not_set_up_is_503_before_azure(client, no_voice):
    r = client.post("/ai/transcribe", content=CLIP, headers=WEBM)
    assert r.status_code == 503 and r.json()["detail"] == "Voice input is not set up on this server yet."


# ---------- the happy path ----------
def test_a_clip_comes_back_as_text(client, voice):
    r = client.post("/ai/transcribe", content=CLIP, headers=WEBM)
    assert r.status_code == 200 and r.json() == {"text": "What was formulary compliance at KR in August?"}
    sent = voice.calls[0]
    assert sent["model"] == "gpt-4o-mini-transcribe" and sent["language"] == "en"
    assert sent["file"] == ("question.webm", CLIP, "audio/webm")          # the codecs part is dropped
    assert sent["prompt"].startswith("Pharmacy analytics for HCG hospitals.")
    assert sent["response_format"] == "json"


@pytest.mark.parametrize("ctype,name", [("audio/ogg;codecs=opus", "question.ogg"), ("audio/mp4", "question.mp4"),
                                        ("audio/wav", "question.wav")])
def test_every_browser_format_is_named_for_azure(client, voice, ctype, name):
    assert client.post("/ai/transcribe", content=CLIP, headers={"Content-Type": ctype}).status_code == 200
    assert voice.calls[0]["file"][0] == name


def test_language_choice(client, voice):
    assert client.post("/ai/transcribe?lang=hi", content=CLIP, headers=WEBM).status_code == 200
    assert voice.calls[-1]["language"] == "hi"
    assert client.post("/ai/transcribe?lang=auto", content=CLIP, headers=WEBM).status_code == 200
    assert "language" not in voice.calls[-1]                              # the model decides
    r = client.post("/ai/transcribe?lang=fr", content=CLIP, headers=WEBM)
    assert r.status_code == 400 and len(voice.calls) == 2


# ---------- refusals ----------
def test_a_non_audio_body_is_415(client, voice):
    for ctype in ("text/plain", "application/json", ""):
        r = client.post("/ai/transcribe", content=CLIP, headers={"Content-Type": ctype} if ctype else {})
        assert r.status_code == 415, ctype
    assert voice.calls == []


def test_too_long_and_too_short(client, voice, monkeypatch):
    monkeypatch.setattr(V, "MAX_AUDIO_BYTES", 2048)
    r = client.post("/ai/transcribe", content=CLIP, headers=WEBM)              # 4 KB > 2 KB
    assert r.status_code == 413 and "under a minute" in r.json()["detail"]
    r = client.post("/ai/transcribe", content=b"\x00" * 100, headers=WEBM)     # < MIN_AUDIO_BYTES
    assert r.status_code == 400 and "Nothing was recorded" in r.json()["detail"]
    assert voice.calls == []


def test_an_azure_failure_is_502_in_plain_words_and_never_logs_the_audio(client, voice, caplog):
    voice.fail = True
    with caplog.at_level(logging.WARNING, logger="app.ai.transcribe"):
        r = client.post("/ai/transcribe", content=CLIP, headers=WEBM)
    assert r.status_code == 502 and r.json()["detail"].startswith("Couldn't turn the recording into text")
    assert "RuntimeError" in caplog.text and "azure said no" not in caplog.text and "test-key" not in caplog.text


def test_silence_is_422(client, voice):
    voice.text = "   "
    r = client.post("/ai/transcribe", content=CLIP, headers=WEBM)
    assert r.status_code == 422 and "No words were heard" in r.json()["detail"]


# ---------- configuration ----------
def test_a_separate_resource_can_hold_the_model(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT", "whisper")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "chat-key")
    monkeypatch.setenv("AZURE_OPENAI_TRANSCRIBE_ENDPOINT", "https://speech-east.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_TRANSCRIBE_API_KEY", "speech-key")
    monkeypatch.delenv("AZURE_OPENAI_TRANSCRIBE_API_VERSION", raising=False)
    s = V.settings()
    assert (s.deployment, s.endpoint, s.key, s.api_version) == \
        ("whisper", "https://speech-east.openai.azure.com", "speech-key", V.TRANSCRIBE_API_VERSION)


def test_the_chat_resource_is_the_default(monkeypatch):
    from app.ai import orchestrator
    monkeypatch.setenv("AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT", "gpt-4o-mini-transcribe")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "chat-key")
    for v in ("AZURE_OPENAI_TRANSCRIBE_ENDPOINT", "AZURE_OPENAI_TRANSCRIBE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    s = V.settings()
    assert (s.endpoint, s.key) == (orchestrator.AZURE_ENDPOINT, "chat-key")


# ---------- the hint list ----------
def test_brand_and_place_words():
    assert V._brand("KEYTRUDA 100MG INJ VIAL") == "KEYTRUDA"
    assert V._brand("DARZALEX FASPRO 1800MG INJ") == "DARZALEX FASPRO"
    assert V._brand("123 ODD NAME") == ""
    assert V._place("HCG KR, Bangalore") == "KR Bangalore"
    assert V._place("HCG Aastha Cancer Centre , Ahmedabad") == "Aastha Ahmedabad"
    assert V._place("HCG Suchirayu") == "Suchirayu"


def test_hint_lists_hospitals_then_drug_brands_by_sales():
    frame = pd.DataFrame([
        ("KEYTRUDA 100MG INJ VIAL", 900.0, "HCG KR, Bangalore", "Onco Drugs"),
        ("KEYTRUDA 25MG INJ VIAL", 200.0, "HCG Suchirayu", "Onco Drugs"),
        ("ERBITUX 100MG INJ", 1000.0, "HCG KR, Bangalore", "Onco Drugs"),
        ("EXAMINATION GLOVES POWDER FREE", 5000.0, "HCG KR, Bangalore", "Consumables"),
        ("PAN 40MG TAB", 10.0, "HCG Suchirayu", "Other Drugs"),
    ], columns=["material_desc", "mrp", "hospital_name", "category"])
    hint = V.hint_from(frame)
    terms = hint.split("Terms: ")[1].rstrip(".").split(", ")
    assert terms[:len(V.BASE_TERMS)] == list(V.BASE_TERMS)
    rest = terms[len(V.BASE_TERMS):]
    # hospitals (sorted), then brands by MRP: KEYTRUDA 1,100 > ERBITUX 1,000 > PAN 10; gloves left out
    assert rest == ["KR Bangalore", "Suchirayu", "KEYTRUDA", "ERBITUX", "PAN"]
    assert "GLOVES" not in hint


def test_hint_is_capped_and_never_empty(monkeypatch):
    assert V.hint_from(None).startswith("Pharmacy analytics for HCG hospitals. Terms: HCG, formulary")
    monkeypatch.setattr(V, "HINT_MAX_CHARS", 40)
    assert len(V.hint_from(None).split("Terms: ")[1]) <= 41
