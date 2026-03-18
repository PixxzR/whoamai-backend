import io
from unittest.mock import patch

import torch
from PIL import Image


def _create_test_image() -> bytes:
    """Crée une image de test en mémoire."""
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _create_face_image() -> bytes:
    """Crée une image de test (pas un vrai visage, détection retournera None)."""
    img = Image.new("RGB", (300, 300), color="beige")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_predict_specialized_no_face(client):
    """Sans vrai visage, doit retourner 422 NO_FACE_DETECTED."""
    image_bytes = _create_test_image()
    response = client.post(
        "/predict/specialized",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "NO_FACE_DETECTED"


def test_predict_multitask_no_face(client):
    image_bytes = _create_test_image()
    response = client.post(
        "/predict/multitask",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "NO_FACE_DETECTED"


def test_predict_transfer_no_face(client):
    image_bytes = _create_test_image()
    response = client.post(
        "/predict/transfer",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "NO_FACE_DETECTED"


def test_predict_invalid_image(client):
    response = client.post(
        "/predict/specialized",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_IMAGE"


def test_predict_image_too_large(client):
    large_bytes = b"x" * (11 * 1024 * 1024)
    response = client.post(
        "/predict/specialized",
        files={"file": ("big.png", large_bytes, "image/png")},
    )
    assert response.status_code == 413
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "IMAGE_TOO_LARGE"


def test_predict_error_response_structure(client):
    """Vérifie la structure complète de la réponse d'erreur."""
    image_bytes = _create_test_image()
    response = client.post(
        "/predict/specialized",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    data = response.json()
    assert "success" in data
    assert "error" in data
    assert "code" in data["error"]
    assert "message" in data["error"]


def _fake_detection(image):
    """Simule une détection de visage pour les tests."""
    return {
        "box": {"x": 50.0, "y": 30.0, "width": 120.0, "height": 150.0},
        "confidence": 0.99,
        "face_tensor": torch.randn(3, 160, 160),
    }


def test_predict_success_response_structure(client):
    """Vérifie la structure complète d'une réponse de succès (demo mode)."""
    image_bytes = _create_test_image()
    with patch("app.main.face_detector") as mock_detector:
        mock_detector.detect = _fake_detection
        response = client.post(
            "/predict/multitask",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    # Vérifie data
    assert "data" in data
    pred = data["data"]
    assert isinstance(pred["age"], (int, float))
    assert pred["gender"] in ("Male", "Female")
    assert 0 <= pred["gender_confidence"] <= 1
    assert pred["ethnicity"] in ("White", "Black", "Asian", "Indian", "Other")
    assert 0 <= pred["ethnicity_confidence"] <= 1
    assert isinstance(pred["ethnicity_class_id"], int)

    # Vérifie metadata
    assert "metadata" in data
    meta = data["metadata"]
    assert meta["strategy"] == "multitask"
    assert meta["demo_mode"] is True
    assert isinstance(meta["inference_time_ms"], (int, float))
    assert meta["face_detection"] is not None
    assert meta["face_detection"]["width"] == 120.0


def test_predict_all_strategies_succeed(client):
    """Les 3 stratégies retournent 200 avec un visage détecté."""
    image_bytes = _create_test_image()
    for strategy in ["specialized", "multitask", "transfer"]:
        with patch("app.main.face_detector") as mock_detector:
            mock_detector.detect = _fake_detection
            response = client.post(
                f"/predict/{strategy}",
                files={"file": ("test.png", image_bytes, "image/png")},
            )
        assert response.status_code == 200, f"{strategy} failed: {response.json()}"
        data = response.json()
        assert data["success"] is True
        assert data["metadata"]["strategy"] == strategy
