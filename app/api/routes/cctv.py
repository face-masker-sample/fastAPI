"""
CCTV 관련 API 라우터
실시간 스트림, 얼굴 탐지 관련 엔드포인트
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/status", response_model=Dict[str, Any])
async def get_cctv_status():
    """CCTV 시스템 상태 조회"""
    try:
        return {
            "status": "active",
            "camera_connected": True,
            "face_detection_enabled": True,
            "message": "CCTV 시스템 정상 작동 중"
        }
    except Exception as e:
        logger.error(f"❌ CCTV 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="CCTV 상태 조회 실패")

@router.post("/start")
async def start_cctv():
    """CCTV 시스템 시작"""
    try:
        return {
            "success": True,
            "message": "CCTV 시스템 시작됨"
        }
    except Exception as e:
        logger.error(f"❌ CCTV 시작 실패: {e}")
        raise HTTPException(status_code=500, detail="CCTV 시작 실패")

@router.post("/stop")
async def stop_cctv():
    """CCTV 시스템 중지"""
    try:
        return {
            "success": True,
            "message": "CCTV 시스템 중지됨"
        }
    except Exception as e:
        logger.error(f"❌ CCTV 중지 실패: {e}")
        raise HTTPException(status_code=500, detail="CCTV 중지 실패") 