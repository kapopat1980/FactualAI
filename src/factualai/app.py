"""Flask REST API layer (Section 3 / Table 1 of the paper).

Seven endpoints: five prediction endpoints (/predict/text, /predict/image,
/predict/audio, /predict/youtube, /rss) and two support endpoints
(/stats, /health). Every prediction endpoint returns the same JSON shape:
final label, overall ensemble confidence, each model's individual
confidence, detected/translated language, and LLM-generated reasoning.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from .classifier import RandomForestNewsClassifier
from .ensemble import classify_with_ensemble
from .llm_reasoner import LLMReasoner
from .preprocessing import preprocess

_START_TIME = time.time()


def create_app(classifier: RandomForestNewsClassifier, llm_reasoner: LLMReasoner | None) -> Flask:
    app = Flask(__name__)
    CORS(app)

    request_count = {"total": 0}

    def _predict_response(raw_text: str) -> dict:
        request_count["total"] += 1
        pre = preprocess(raw_text)
        rf_score = classifier.rf_fake_score(pre.clean_text)
        result = classify_with_ensemble(pre.clean_text, rf_score, llm_reasoner)
        return {
            "label": result.label,
            "confidence": round(result.confidence, 4),
            "ensemble_fake_score": round(result.ensemble_fake_score, 4),
            "rf_confidence": round(result.rf_fake_score, 4),
            "llm_confidence": round(result.llm_fake_score, 4) if result.llm_fake_score is not None else None,
            "reasoning": result.llm_justification,
            "original_language": pre.original_language,
            "translated": pre.translated,
        }

    @app.route("/predict/text", methods=["POST"])
    def predict_text():
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text.strip():
            return jsonify({"error": "No text provided."}), 400
        return jsonify(_predict_response(text))

    @app.route("/predict/image", methods=["POST"])
    def predict_image():
        from .ocr import extract_text_from_image, OCR_PREVIEW_CHARS

        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400
        image_file = request.files["image"]
        with tempfile.NamedTemporaryFile(suffix=Path(image_file.filename).suffix, delete=False) as tmp:
            image_file.save(tmp.name)
            extracted_text = extract_text_from_image(tmp.name)
        response = _predict_response(extracted_text)
        response["extracted_text_preview"] = extracted_text[:OCR_PREVIEW_CHARS]
        return jsonify(response)

    @app.route("/predict/audio", methods=["POST"])
    def predict_audio():
        from .asr import transcribe_audio

        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided."}), 400
        audio_file = request.files["audio"]
        with tempfile.NamedTemporaryFile(suffix=Path(audio_file.filename).suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            transcript = transcribe_audio(tmp.name)
        response = _predict_response(transcript)
        response["transcript"] = transcript
        return jsonify(response)

    @app.route("/predict/youtube", methods=["POST"])
    def predict_youtube():
        from .youtube import transcribe_youtube_video

        data = request.get_json(force=True)
        url = data.get("url", "")
        if not url:
            return jsonify({"error": "No YouTube URL provided."}), 400
        transcript = transcribe_youtube_video(url)
        response = _predict_response(transcript)
        response["transcript"] = transcript
        return jsonify(response)

    @app.route("/rss", methods=["GET"])
    def rss_feed():
        from .rss import classify_live_feed

        items = classify_live_feed(classifier)
        return jsonify(
            [
                {
                    "source": item.source,
                    "title": item.title,
                    "summary": item.summary,
                    "label": item.label,
                    "confidence": round(item.confidence, 4),
                }
                for item in items
            ]
        )

    @app.route("/stats", methods=["GET"])
    def stats():
        return jsonify({"total_requests": request_count["total"], "uptime_seconds": round(time.time() - _START_TIME, 1)})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    return app
