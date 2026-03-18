from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "FaceSense API"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://10.0.2.2:8000",  # Android emulator
    ]

    models_dir: str = "./models"
    device: str = "cpu"

    max_image_size_mb: int = 10
    inference_timeout_sec: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
