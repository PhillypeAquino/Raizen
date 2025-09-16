from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

class Settings(BaseModel):
    api_base: str = os.getenv("API_BASE", "").rstrip("/")
    username: str = os.getenv("API_USERNAME", "")
    password: str = os.getenv("API_PASSWORD", "")
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data")).resolve()

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
(settings.data_dir / "raw").mkdir(exist_ok=True)
(settings.data_dir / "processed").mkdir(exist_ok=True)