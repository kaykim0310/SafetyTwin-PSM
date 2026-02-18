"""
PSM-SafetyTwin SVC-01: pid-parser 설정
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모든 설정을 환경 변수로 관리합니다.
.env 파일 또는 NCP 환경 변수에서 읽어옵니다.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """pid-parser 서비스 설정"""

    # ── 서비스 기본 ──
    SERVICE_NAME: str = "pid-parser"
    SERVICE_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── D-Fine 모델 설정 ──
    DFINE_MODEL_NAME: str = "DFine-L"               # L(Large) 모델 기본
    DFINE_WEIGHTS_PATH: str = "models/dfine/dfine_l_obj365.pth"  # 사전학습 가중치
    DFINE_CONFIDENCE_THRESHOLD: float = 0.5          # 심볼 감지 최소 신뢰도
    DFINE_DEVICE: str = "cuda"                       # GPU 사용 (cpu 전환 가능)
    DFINE_INPUT_SIZE: int = 640                      # 입력 이미지 크기

    # ── P&ID 심볼 클래스 ──
    # Digitize-PID 데이터셋 기반 + PSM 특화 심볼 추가
    PID_SYMBOL_CLASSES: list[str] = [
        # 밸브류 (Valves)
        "gate_valve", "globe_valve", "ball_valve", "check_valve",
        "butterfly_valve", "relief_valve", "control_valve",
        "solenoid_valve", "needle_valve",
        # 장치류 (Equipment)
        "pump", "compressor", "tank", "vessel", "reactor",
        "heat_exchanger", "column", "drum", "filter",
        # 계장류 (Instruments)
        "pressure_indicator", "temperature_indicator",
        "flow_indicator", "level_indicator",
        "pressure_transmitter", "temperature_transmitter",
        "flow_transmitter", "level_transmitter",
        "control_loop", "alarm",
        # 배관류 (Piping)
        "pipe_line", "reducer", "tee", "elbow",
        "flange", "spectacle_blind",
        # 안전장치 (Safety - PSM 특화)
        "rupture_disc", "flame_arrestor", "safety_shower",
        "emergency_shutoff",
    ]

    # ── PaddleOCR 설정 ──
    OCR_LANG: str = "korean"                         # 한국어 도면 지원
    OCR_USE_GPU: bool = True

    # ── 이미지 전처리 설정 ──
    IMAGE_MAX_SIZE: int = 4096                       # 최대 이미지 크기 (px)
    IMAGE_SEGMENT_SIZE: int = 640                    # 분할 세그먼트 크기
    IMAGE_SEGMENT_OVERLAP: int = 64                  # 세그먼트 겹침 영역

    # ── 데이터베이스 ──
    POSTGRES_URL: str = "postgresql+asyncpg://psm:psm@localhost:5432/pid_parser"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "psm-safetytwin"

    # ── 파일 저장 ──
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    MAX_UPLOAD_SIZE_MB: int = 100                    # 최대 업로드 크기

    # ── 공공데이터 API ──
    DATA_GO_KR_API_KEY: Optional[str] = None         # 공공데이터포털 인증키
    KOSHA_API_BASE_URL: str = "https://msds.kosha.or.kr"

    # ── Redis / Celery (배치 처리) ──
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── 로깅 ──
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 싱글톤 설정 인스턴스
settings = Settings()
