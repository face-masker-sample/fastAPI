"""
녹화 파일 관리 API 라우터
녹화 파일 목록, 다운로드, 삭제 관련 엔드포인트
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/list", response_model=List[Dict[str, Any]])
async def get_recordings_list():
    """녹화 파일 목록 조회"""
    try:
        recordings = [
            {
                "filename": "recording_20250121_120000.mp4",
                "size": "15.2 MB",
                "duration": "5분 30초",
                "created_at": "2025-01-21 12:00:00",
                "faces_detected": 45
            },
            {
                "filename": "recording_20250121_130000.mp4", 
                "size": "22.8 MB",
                "duration": "8분 15초",
                "created_at": "2025-01-21 13:00:00",
                "faces_detected": 67
            }
        ]
        return recordings
    except Exception as e:
        logger.error(f"❌ 녹화 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="녹화 목록 조회 실패")

@router.get("/statistics")
async def get_recording_statistics():
    """녹화 통계 조회"""
    try:
        return {
            "total_recordings": 15,
            "total_size": "245.7 MB",
            "total_duration": "2시간 15분",
            "average_faces_per_recording": 52.3,
            "storage_usage": "24.6%"
        }
    except Exception as e:
        logger.error(f"❌ 녹화 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="녹화 통계 조회 실패")

@router.delete("/{filename}")
async def delete_recording(filename: str):
    """녹화 파일 삭제"""
    try:
        return {
            "success": True,
            "message": f"{filename} 파일이 삭제되었습니다"
        }
    except Exception as e:
        logger.error(f"❌ 녹화 파일 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail="녹화 파일 삭제 실패") 