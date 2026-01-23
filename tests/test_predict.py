import io

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
    """Sans vrai visage, face_detected doit être False."""
    image_bytes = _create_test_image()
    response = client.post(
        "/predict/specialized",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["strategy"] == "specialized"
    assert data["face_detected"] is False


def test_predict_multitask_no_face(client):
    image_bytes = _create_test_image()
    response = client.post(
        "/predict/multitask",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["face_detected"] is False


def test_predict_transfer_no_face(client):
    image_bytes = _create_test_image()
    response = client.post(
        "/predict/transfer",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["face_detected"] is False


def test_predict_invalid_image(client):
    response = client.post(
        "/predict/specialized",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_image_too_large(client):
    # Créer un payload > 10MB
    large_bytes = b"x" * (11 * 1024 * 1024)
    response = client.post(
        "/predict/specialized",
        files={"file": ("big.png", large_bytes, "image/png")},
    )
    assert response.status_code == 413
