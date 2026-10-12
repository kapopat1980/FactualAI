"""LLM-based contextual reasoning via a locally hosted LLaMA 3.2 model served
through Ollama (Section 4.5 of the paper).

The first 1,500 characters of an article are sent to the LLM with a
structured prompt requesting a JSON response containing a label (Fake/Real),
a confidence score (0.5-1.0), and a short plain-language justification. The
call has a 30-second timeout; if Ollama is unreachable or the timeout is hit,
`LLMReasoner.classify` raises `LLMUnavailableError`, which callers (see
`ensemble.py`) catch to fall back to the Random Forest prediction alone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import requests

MAX_INPUT_CHARS = 1500
TIMEOUT_SECONDS = 30
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
import os

# Deployed default; LLaMA 3.1 8B ("llama3.1:8b") performed far better in the revised evaluation (Table 7).
DEFAULT_MODEL = os.environ.get("FACTUALAI_LLM_MODEL", "llama3.2:1b")

_PROMPT_TEMPLATE = """You are a fact-checking assistant. Read the following news \
article excerpt and decide whether it is more likely to be real or fake news.

Respond with ONLY a JSON object of the form:
{{"label": "Fake" | "Real", "confidence": <float between 0.5 and 1.0>, \
"justification": "<one or two sentence plain-language explanation>"}}

Article excerpt:
\"\"\"{article}\"\"\"
"""


class LLMUnavailableError(RuntimeError):
    """Raised when the Ollama service cannot be reached within the timeout,
    or returns a response that cannot be parsed as the expected JSON schema."""


@dataclass
class LLMVerdict:
    label: str  # "Fake" or "Real" or "Unknown"
    confidence: float  # 0.5-1.0
    justification: str


class LLMReasoner:
    """Client for the locally hosted LLaMA 3.2 model via Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = TIMEOUT_SECONDS,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout

    def classify(self, article_text: str) -> LLMVerdict:
        excerpt = article_text[:MAX_INPUT_CHARS]
        prompt = _PROMPT_TEMPLATE.format(article=excerpt)
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMUnavailableError(f"Ollama request failed: {exc}") from exc

        try:
            raw_output = response.json()["response"]
            parsed = json.loads(raw_output)
            label = parsed.get("label", "Unknown")
            confidence = float(parsed.get("confidence", 0.5))
            justification = parsed.get("justification", "")
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise LLMUnavailableError(f"Could not parse Ollama response: {exc}") from exc

        return LLMVerdict(label=label, confidence=confidence, justification=justification)
