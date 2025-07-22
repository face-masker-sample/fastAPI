"""
FastAPI 메인 애플리케이션
무인관제모텔 CCTV 얼굴 마스킹 시스템의 백엔드 서버
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import uvicorn
import asyncio
import json
from datetime import datetime
import logging

from app.core.config import settings
from app.api.routes import cctv, control, recordings
from app.services.face_masking_service import FaceMaskingService
from app.services.websocket_manager import WebSocketManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="CCTV Face Masking Control System",
    description="무인관제모텔 CCTV 얼굴 마스킹 시스템",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# 서비스 초기화
face_masking_service = FaceMaskingService()
websocket_manager = WebSocketManager()

# API 라우터 등록
app.include_router(cctv.router, prefix="/api/cctv", tags=["CCTV"])
app.include_router(control.router, prefix="/api/control", tags=["Control"])
app.include_router(recordings.router, prefix="/api/recordings", tags=["Recordings"])

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작시 초기화"""
    logger.info("🚀 CCTV Face Masking System 시작")
    logger.info(f"📊 설정: {settings.model_dump()}")
    
    # 얼굴 마스킹 서비스 초기화
    await face_masking_service.initialize()
    logger.info("✅ Face Masking Service 초기화 완료")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료시 정리"""
    logger.info("🛑 CCTV Face Masking System 종료")
    await face_masking_service.cleanup()
    logger.info("✅ 시스템 정리 완료")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """관제 대시보드 메인 페이지"""
    return templates.TemplateResponse(
        "dashboard.html", 
        {
            "request": request,
            "title": "CCTV 관제 대시보드",
            "version": "1.0.0"
        }
    )

@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """실시간 CCTV 스트림 WebSocket"""
    await websocket_manager.connect(websocket)
    logger.info("🔌 WebSocket 연결: 실시간 스트림")
    
    try:
        # 실시간 얼굴 마스킹 스트림 시작
        async for frame_data in face_masking_service.get_masked_stream():
            await websocket_manager.send_frame(websocket, frame_data)
            
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket 연결 해제: 실시간 스트림")
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket 스트림 오류: {e}")
        websocket_manager.disconnect(websocket)

@app.websocket("/ws/control")
async def websocket_control_endpoint(websocket: WebSocket):
    """관제 시스템 제어 WebSocket"""
    await websocket_manager.connect(websocket)
    logger.info("🔌 WebSocket 연결: 시스템 제어")
    
    try:
        while True:
            # 클라이언트로부터 제어 명령 수신
            data = await websocket.receive_text()
            command_data = json.loads(data)
            
            # 명령 처리
            response = await handle_control_command(command_data)
            
            # 응답 전송
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket 연결 해제: 시스템 제어")
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket 제어 오류: {e}")
        websocket_manager.disconnect(websocket)

async def handle_control_command(command_data: dict) -> dict:
    """제어 명령 처리"""
    command = command_data.get("command")
    
    if command == "start_recording":
        success = await face_masking_service.start_recording()
        return {
            "command": command,
            "success": success,
            "message": "녹화 시작됨" if success else "녹화 시작 실패",
            "timestamp": datetime.now().isoformat()
        }
        
    elif command == "stop_recording":
        success = await face_masking_service.stop_recording()
        return {
            "command": command,
            "success": success,
            "message": "녹화 중지됨" if success else "녹화 중지 실패",
            "timestamp": datetime.now().isoformat()
        }
        
    elif command == "take_screenshot":
        filename = await face_masking_service.take_screenshot()
        return {
            "command": command,
            "success": filename is not None,
            "filename": filename,
            "message": f"스크린샷 저장: {filename}" if filename else "스크린샷 실패",
            "timestamp": datetime.now().isoformat()
        }
        
    elif command == "get_stats":
        stats = await face_masking_service.get_statistics()
        return {
            "command": command,
            "success": True,
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    else:
        return {
            "command": command,
            "success": False,
            "message": f"알 수 없는 명령: {command}",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 

## python realtime_face_masking_emoji.py
## python -m app.main
## uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
## conda activate face-masker