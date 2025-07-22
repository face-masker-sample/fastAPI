"""
WebSocket 연결 관리 서비스
실시간 스트리밍 및 제어를 위한 WebSocket 연결 관리
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

class WebSocketManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        """연결 관리자 초기화"""
        self.active_connections: List[WebSocket] = []
        self.connection_count = 0
        
    async def connect(self, websocket: WebSocket):
        """WebSocket 연결"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_count += 1
        logger.info(f"🔌 WebSocket 연결됨. 총 연결: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """WebSocket 연결 해제"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"🔌 WebSocket 연결 해제됨. 총 연결: {len(self.active_connections)}")
            
    async def send_frame(self, websocket: WebSocket, frame_data: Dict[str, Any]):
        """개별 WebSocket에 프레임 전송"""
        try:
            await websocket.send_text(json.dumps(frame_data))
        except Exception as e:
            logger.error(f"❌ 프레임 전송 실패: {e}")
            self.disconnect(websocket)
            
    async def broadcast_frame(self, frame_data: Dict[str, Any]):
        """모든 연결된 WebSocket에 프레임 브로드캐스트"""
        if not self.active_connections:
            return
            
        disconnected = []
        
        for websocket in self.active_connections:
            try:
                await websocket.send_text(json.dumps(frame_data))
            except Exception as e:
                logger.error(f"❌ 브로드캐스트 실패: {e}")
                disconnected.append(websocket)
                
        # 연결 해제된 WebSocket 정리
        for websocket in disconnected:
            self.disconnect(websocket)
            
    async def send_control_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """제어 메시지 전송"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"❌ 제어 메시지 전송 실패: {e}")
            self.disconnect(websocket)
            
    async def broadcast_control_message(self, message: Dict[str, Any]):
        """모든 연결에 제어 메시지 브로드캐스트"""
        if not self.active_connections:
            return
            
        disconnected = []
        
        for websocket in self.active_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"❌ 제어 메시지 브로드캐스트 실패: {e}")
                disconnected.append(websocket)
                
        # 연결 해제된 WebSocket 정리
        for websocket in disconnected:
            self.disconnect(websocket)
            
    def get_connection_count(self) -> int:
        """현재 연결 수 반환"""
        return len(self.active_connections)
        
    def get_statistics(self) -> Dict[str, Any]:
        """WebSocket 통계 반환"""
        return {
            "active_connections": len(self.active_connections),
            "total_connections": self.connection_count,
            "connection_ids": [id(ws) for ws in self.active_connections]
        } 