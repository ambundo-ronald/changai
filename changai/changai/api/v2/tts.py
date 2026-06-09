import re
import base64
import boto3
import frappe
from typing import Any, Dict, Optional
from changai.changai.api.v2.schema_utils import ChangAIConfig

_POLLY_CLIENT = None


def get_polly_client(config):
    global _POLLY_CLIENT

    if _POLLY_CLIENT is None:
        _POLLY_CLIENT = boto3.client(
            "polly",
            aws_access_key_id=(config.get("aws_access_key_id") or "").strip(),
            aws_secret_access_key=(config.get("aws_secret_access_key") or "").strip(),
            region_name=(config.get("aws_region") or "us-east-1"),
        )
    return _POLLY_CLIENT

def build_ssml(text: str) -> str:
    parts = []
    current = []
    current_lang = None

    for token in text.split():
        lang = "ar-AE" if re.search(r'[\u0600-\u06FF]', token) else "en-US"

        if current_lang is None:
            current_lang = lang

        if lang != current_lang:
            parts.append(
                f'<lang xml:lang="{current_lang}">{" ".join(current)}</lang>'
            )
            current = [token]
            current_lang = lang
        else:
            current.append(token)

    if current:
        parts.append(
            f'<lang xml:lang="{current_lang}">{" ".join(current)}</lang>'
        )

    return "<speak>" + " ".join(parts) + "</speak>"
@frappe.whitelist(allow_guest=False)
def synthesize_tts(text: str, voice_id: Optional[str] = None) -> Dict[str, Any]:
    config = ChangAIConfig.get()
    if not bool(config.get("enable_voice_chat")):
        return {"ok": False, "error": "Voice chat is disabled in settings.", "provider": "browser"}
    aws_access_key_id = (config.get("aws_access_key_id") or "").strip()
    aws_secret_access_key = (config.get("aws_secret_access_key") or "").strip()
    if not aws_access_key_id or not aws_secret_access_key:
        return {"ok": False, "error": "AWS Polly credentials are missing.", "provider": "browser"}
    cleaned_text = re.sub(r"<[^>]*>", " ", text or "")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    if not cleaned_text:
        return {"ok": False, "error": "Text is empty.", "provider": "browser"}

    if len(cleaned_text) > 2500:
        cleaned_text = cleaned_text[:2500]

    try:
        polly_client = get_polly_client(config)
        voice = (voice_id or config.get("polly_voice_id") or "Zayd").strip() or "Zayd"
        ssml_text = build_ssml(cleaned_text)
        response = polly_client.synthesize_speech(
    Text=ssml_text,
    OutputFormat="mp3",
    VoiceId="Zayd",
    Engine="neural",
    TextType="ssml",
)
        stream = response.get("AudioStream")
        if stream is None:
            return {"ok": False, "error": "Polly did not return audio stream.", "provider": "browser"}

        audio_bytes = stream.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {
            "ok": True,
            "provider": "polly",
            "mime_type": "audio/mpeg",
            "audio_base64": audio_base64,
            "voice_id": voice,
        }
    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "ChangAI Polly TTS Error")
        return {"ok": False, "error": str(e), "provider": "browser"}

