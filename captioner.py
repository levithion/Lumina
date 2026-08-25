"""Vision-language captioning and safety classification for ingestion.

Runs only inside ingestion pipelines (sync_memes.py, backfill_v2.py); the
serving apps never load this model, keeping Streamlit Cloud memory flat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from PIL import Image

from config import CAPTION_ENABLED, CAPTION_MODEL_NAME


@dataclass
class CaptionResult:
    caption: str = ""
    is_sensitive: bool = False
    raw: str = ""


_CAPTION_PROMPT = (
    "Describe this meme in one concise sentence, including any visible text. "
    "Then on a new line write exactly 'Sensitive: yes' if the image contains "
    "pornography, extreme violence, or hate symbols, otherwise 'Sensitive: no'."
)


class MemeCaptioner:
    """Small VLM that turns meme images into searchable captions plus a safety flag."""

    def __init__(self, model_name: str = CAPTION_MODEL_NAME, device: str | None = None) -> None:
        if not CAPTION_ENABLED:
            raise RuntimeError("CAPTION_ENABLED=0: vision captioning is disabled")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._device = device or (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=dtype).to(self._device)
        self._model.eval()

    def caption_image(self, image: Image.Image) -> CaptionResult:
        import torch

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _CAPTION_PROMPT},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self._processor(images=image.convert("RGB"), text=prompt, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=120, do_sample=False)
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _parse_output(text)


def _parse_output(text: str) -> CaptionResult:
    # Decoded output can include the prompt; keep only what follows the
    # assistant marker. The marker's trailing colon survives the split, so
    # lines may start with punctuation.
    body = re.split(r"\bassistant\b", text, flags=re.IGNORECASE)[-1]
    sensitive = False
    caption_lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^[^\w\s]*sensitive\s*:\s*(yes|no|true|false)\b", stripped, flags=re.IGNORECASE)
        if match:
            sensitive = match.group(1).lower() in ("yes", "true")
        elif not caption_lines:
            caption_lines.append(stripped)
    return CaptionResult(caption=" ".join(caption_lines)[:400].lstrip(":").strip(), is_sensitive=sensitive, raw=text.strip())


class DisabledCaptioner:
    """Drop-in stand-in used when CAPTION_ENABLED=0; keeps ingestion running."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def caption_image(self, image: Image.Image) -> CaptionResult:
        return CaptionResult()


def build_captioner(model_name: str = CAPTION_MODEL_NAME, device: str | None = None):
    """Return a MemeCaptioner, or DisabledCaptioner when captioning is off."""
    if not CAPTION_ENABLED:
        return DisabledCaptioner()
    try:
        return MemeCaptioner(model_name=model_name, device=device)
    except Exception as exc:
        # Missing weights / no backend should never block ingestion entirely,
        # but the fallback must be loud — silent degradation is how empty
        # captions went unnoticed across scheduled runs.
        print(f"WARNING: captioning disabled, {model_name} failed to load ({exc})")
        return DisabledCaptioner()
