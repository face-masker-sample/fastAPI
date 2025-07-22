"""
실시간 얼굴 마스킹 서비스
CCTV 스트림에서 얼굴을 탐지하고 마스킹 처리하는 핵심 서비스
"""

import cv2
import numpy as np
import asyncio
import base64
import time
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

class FaceMaskingService:
    """실시간 얼굴 마스킹 서비스"""
    
    def __init__(self):
        """서비스 초기화"""
        self.model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_recording: bool = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.stats = {
            "total_frames": 0,
            "total_faces_detected": 0,
            "average_fps": 0.0,
            "last_detection_time": None,
            "recording_status": False,
            "system_start_time": datetime.now().isoformat()
        }
        
    async def initialize(self):
        """서비스 초기화"""
        logger.info("🚀 얼굴 마스킹 서비스 초기화 시작")
        
        # YOLO 모델 로드
        await self._load_model()
        
        # 웹캠 초기화
        await self._setup_webcam()
        
        logger.info("✅ 얼굴 마스킹 서비스 초기화 완료")
        
    async def _load_model(self):
        """YOLO 얼굴 탐지 모델 로드"""
        try:
            logger.info("🤖 YOLOv8-Face 모델 다운로드 중...")
            model_path = hf_hub_download(
                repo_id=settings.YOLO_MODEL_REPO,
                filename=settings.YOLO_MODEL_FILE
            )
            self.model = YOLO(model_path)
            logger.info("✅ YOLOv8-Face 모델 로드 성공")
            
        except Exception as e:
            logger.warning(f"⚠️ YOLOv8-Face 모델 로드 실패: {e}")
            logger.info("🔄 기본 YOLO 모델로 대체")
            self.model = YOLO('yolov8n.pt')
            
    async def _setup_webcam(self):
        """웹캠 초기화"""
        try:
            self.cap = cv2.VideoCapture(settings.WEBCAM_INDEX)
            
            # 웹캠 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.WEBCAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.WEBCAM_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, settings.WEBCAM_FPS)
            
            if not self.cap.isOpened():
                raise Exception("웹캠을 열 수 없습니다")
                
            logger.info("📹 웹캠 초기화 성공")
            
        except Exception as e:
            logger.error(f"❌ 웹캠 초기화 실패: {e}")
            raise
            
    def _create_smooth_blur(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """빠른 이모지 마스킹 (성능 최적화)"""
        h, w = image.shape[:2]
        
        # 경계 확인
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return image
        
        # 얼굴 크기 계산
        face_w, face_h = x2 - x1, y2 - y1
        face_size = min(face_w, face_h)
        
        # 이모지 중심점 계산
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 이모지 크기 (얼굴보다 약간 크게)
        emoji_radius = int(face_size * 0.6)
        
        # 스마일 이모지 그리기 (빠른 버전)
        self._draw_fast_smile_emoji(image, center_x, center_y, emoji_radius)
        
        return image
    
    def _draw_fast_smile_emoji(self, image, center_x, center_y, radius):
        """고속 스마일 이모지 그리기"""
        # 노란색 배경 원
        cv2.circle(image, (center_x, center_y), radius, (0, 215, 255), -1)  # BGR: 노란색
        
        # 검은색 테두리
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 0), 3)
        
        # 눈 그리기
        eye_offset_x = radius // 3
        eye_offset_y = radius // 4
        eye_radius = radius // 8
        
        # 왼쪽 눈
        cv2.circle(image, (center_x - eye_offset_x, center_y - eye_offset_y), eye_radius, (0, 0, 0), -1)
        
        # 오른쪽 눈
        cv2.circle(image, (center_x + eye_offset_x, center_y - eye_offset_y), eye_radius, (0, 0, 0), -1)
        
        # 입 그리기 (스마일)
        mouth_y = center_y + radius // 4
        mouth_width = radius // 2
        mouth_height = radius // 3
        
        # 스마일 호 그리기
        cv2.ellipse(image, (center_x, mouth_y), (mouth_width, mouth_height), 0, 0, 180, (0, 0, 0), 4)
        
    def _detect_faces(self, frame: np.ndarray) -> list:
        """얼굴 탐지"""
        if self.model is None:
            return []
            
        # 이미지 품질 향상
        enhanced = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # YOLO 탐지
        results = self.model(
            enhanced,
            conf=settings.FACE_CONFIDENCE_THRESHOLD,
            iou=settings.FACE_IOU_THRESHOLD,
            imgsz=settings.INPUT_IMAGE_SIZE,
            verbose=False
        )
        
        faces = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # 최소 크기 체크
                    w, h = x2 - x1, y2 - y1
                    if w >= 20 and h >= 20:  # 20x20 픽셀 이상
                        faces.append((x1, y1, x2, y2, conf))
                        
        return faces
        
    async def get_masked_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """마스킹된 실시간 스트림 제공"""
        if self.cap is None:
            logger.error("❌ 웹캠이 초기화되지 않음")
            return
            
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("⚠️ 웹캠에서 프레임을 읽을 수 없음")
                await asyncio.sleep(0.1)
                continue
                
            # 좌우 반전 (거울 효과)
            frame = cv2.flip(frame, 1)
            
            # 얼굴 탐지
            faces = self._detect_faces(frame)
            
            # 얼굴 마스킹
            for x1, y1, x2, y2, conf in faces:
                frame = self._create_smooth_blur(frame, x1, y1, x2, y2)
                
            # 녹화 중이면 비디오 파일에 저장
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(frame)
                
            # FPS 계산
            fps_counter += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                self.stats["average_fps"] = current_fps
                fps_counter = 0
                start_time = time.time()
            else:
                current_fps = self.stats["average_fps"]
                
            # 통계 업데이트
            self.stats["total_frames"] += 1
            self.stats["total_faces_detected"] += len(faces)
            if len(faces) > 0:
                self.stats["last_detection_time"] = datetime.now().isoformat()
                
            # 정보 표시
            info_text = f'FPS: {current_fps:.1f} | Faces: {len(faces)}'
            if self.is_recording:
                info_text += " | REC"
                
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                       
            # 프레임을 JPEG로 인코딩 (성능 최적화)
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, settings.STREAM_QUALITY,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1  # JPEG 최적화 활성화
            ]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 데이터 반환
            yield {
                "frame": frame_base64,
                "faces_count": len(faces),
                "fps": current_fps,
                "timestamp": datetime.now().isoformat(),
                "recording": self.is_recording
            }
            
            # FPS 조절
            await asyncio.sleep(1.0 / settings.STREAM_FPS)
            
    async def start_recording(self) -> bool:
        """녹화 시작"""
        if self.is_recording:
            logger.warning("⚠️ 이미 녹화 중입니다")
            return False
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{settings.RECORDING_DIR}/recording_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*settings.RECORDING_FORMAT)
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, settings.STREAM_FPS, 
                (settings.WEBCAM_WIDTH, settings.WEBCAM_HEIGHT)
            )
            
            self.is_recording = True
            self.stats["recording_status"] = True
            logger.info(f"🎬 녹화 시작: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 녹화 시작 실패: {e}")
            return False
            
    async def stop_recording(self) -> bool:
        """녹화 중지"""
        if not self.is_recording:
            logger.warning("⚠️ 녹화 중이 아닙니다")
            return False
            
        try:
            self.is_recording = False
            self.stats["recording_status"] = False
            
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                
            logger.info("🛑 녹화 중지됨")
            return True
            
        except Exception as e:
            logger.error(f"❌ 녹화 중지 실패: {e}")
            return False
            
    async def take_screenshot(self) -> Optional[str]:
        """스크린샷 촬영"""
        if self.cap is None:
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                return None
                
            # 좌우 반전
            frame = cv2.flip(frame, 1)
            
            # 얼굴 탐지 및 마스킹
            faces = self._detect_faces(frame)
            for x1, y1, x2, y2, conf in faces:
                frame = self._create_smooth_blur(frame, x1, y1, x2, y2)
                
            # 파일 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{settings.SCREENSHOT_DIR}/screenshot_{timestamp}.jpg"
            
            cv2.imwrite(filename, frame)
            logger.info(f"📸 스크린샷 저장: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ 스크린샷 실패: {e}")
            return None
            
    async def get_statistics(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        return self.stats.copy()
        
    async def cleanup(self):
        """서비스 정리"""
        logger.info("🧹 얼굴 마스킹 서비스 정리 시작")
        
        # 녹화 중지
        if self.is_recording:
            await self.stop_recording()
            
        # 웹캠 해제
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        logger.info("✅ 얼굴 마스킹 서비스 정리 완료") 