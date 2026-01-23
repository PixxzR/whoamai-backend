import io

from PIL import Image


def _create_test_image() -> bytes:
    """Crée une image de test en mémoire."""
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def test_predict_specialized_not_implemented(client):
    image_bytes = _create_test_image()
    response = client.post(
        "/predict/specialized",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 501


def test_predict_multitask_not_implemented(client):
    image_bytes = _create_test_image()
    response = client.post(
        "/predict/multitask",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 501


def test_predict_transfer_not_implemented(client):
    image_bytes = _create_test_image()
    response = client.post(
        "/predict/transfer",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 501
