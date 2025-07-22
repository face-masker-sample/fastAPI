"""
애플리케이션 설정 관리
환경 변수 및 구성 설정
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # 프로젝트 설정
    PROJECT_NAME: str = "CCTV Face Masking Control System"
    PROJECT_VERSION: str = "1.0.0"
    
    # 얼굴 마스킹 모델 설정
    YOLO_MODEL_REPO: str = "arnabdhar/YOLOv8-Face-Detection"
    YOLO_MODEL_FILE: str = "model.pt"
    FACE_CONFIDENCE_THRESHOLD: float = 0.05
    FACE_IOU_THRESHOLD: float = 0.4
    INPUT_IMAGE_SIZE: int = 640
    
    # 웹캠 설정
    WEBCAM_INDEX: int = 0
    WEBCAM_WIDTH: int = 1280
    WEBCAM_HEIGHT: int = 720
    WEBCAM_FPS: int = 30
    
    # 스트리밍 설정
    STREAM_FPS: int = 25
    STREAM_QUALITY: int = 80  # JPEG 품질 (1-100)
    
    # 녹화 설정
    RECORDING_DIR: str = "recordings"
    RECORDING_FORMAT: str = "mp4v"
    SCREENSHOT_DIR: str = "screenshots"
    
    # 블러 처리 설정
    BLUR_STRENGTH: int = 25
    BLUR_EXPAND_RATIO_W: float = 0.3
    BLUR_EXPAND_RATIO_H: float = 0.4
    BLUR_FADE_SIZE: int = 20
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 보안 설정
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # 데이터베이스 설정 (향후 확장용)
    DATABASE_URL: Optional[str] = None
    REDIS_URL: Optional[str] = None
    
    # 알림 설정 (향후 확장용)
    ENABLE_EMAIL_ALERTS: bool = False
    SMTP_SERVER: Optional[str] = None
    SMTP_PORT: Optional[int] = None
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 디렉토리 생성
        os.makedirs(self.RECORDING_DIR, exist_ok=True)
        os.makedirs(self.SCREENSHOT_DIR, exist_ok=True)

# 전역 설정 인스턴스
settings = Settings()

# 개발/프로덕션 환경별 설정
class DevelopmentSettings(Settings):
    """개발 환경 설정"""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"

class ProductionSettings(Settings):
    """프로덕션 환경 설정"""
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    
def get_settings() -> Settings:
    """환경에 따른 설정 반환"""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionSettings()
    else:
        return DevelopmentSettings() 