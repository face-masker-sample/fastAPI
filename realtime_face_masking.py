"""
실시간 웹캠 얼굴 마스킹 프로그램
YOLOv8-Face 모델을 사용하여 실시간으로 얼굴을 탐지하고 블러 처리
"""

import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os
import time

class RealTimeFaceMasking:
    def __init__(self):
        """실시간 얼굴 마스킹 시스템 초기화"""
        print("🚀 실시간 얼굴 마스킹 시스템 초기화 중...")
        
        # YOLOv8-Face 모델 로드
        self.load_model()
        
        # 웹캠 초기화
        self.setup_webcam()
        
        print("✅ 시스템 준비 완료!")
        print("📹 웹캠 시작: ESC 키로 종료")
    
    def load_model(self):
        """YOLOv8-Face 모델 로드"""
        try:
            print("🤖 YOLOv8-Face 모델 다운로드 중...")
            model_path = hf_hub_download(
                repo_id="arnabdhar/YOLOv8-Face-Detection", 
                filename="model.pt"
            )
            self.model = YOLO(model_path)
            print("✅ 모델 로드 완료!")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("🔄 기본 YOLO 모델로 대체...")
            self.model = YOLO('yolov8n.pt')
    
    def setup_webcam(self):
        """웹캠 초기화"""
        self.cap = cv2.VideoCapture(0)  # 기본 웹캠
        
        # 웹캠 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise Exception("웹캠을 열 수 없습니다!")
    
    def create_smooth_blur(self, image, x1, y1, x2, y2, blur_strength=25):
        """자연스러운 블러 마스킹"""
        h, w = image.shape[:2]
        
        # 경계 확인
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return image
        
        # 얼굴 영역 확장 (고개 숙임 대비)
        face_w, face_h = x2 - x1, y2 - y1
        expand_w = int(face_w * 0.3)
        expand_h = int(face_h * 0.4)
        
        # 확장된 영역
        exp_x1 = max(0, x1 - expand_w)
        exp_y1 = max(0, y1 - expand_h)
        exp_x2 = min(w, x2 + expand_w)
        exp_y2 = min(h, y2 + expand_h)
        
        # 얼굴 영역 추출
        face_region = image[exp_y1:exp_y2, exp_x1:exp_x2].copy()
        
        if face_region.size == 0:
            return image
        
        # 강한 가우시안 블러
        blurred = cv2.GaussianBlur(face_region, (blur_strength*2+1, blur_strength*2+1), blur_strength)
        
        # 추가 블러 (더 뿌옇게)
        extra_blur = cv2.bilateralFilter(blurred, 15, 100, 100)
        
        # 마스크 생성 (가장자리 페이드)
        mask = np.ones((exp_y2-exp_y1, exp_x2-exp_x1, 3), dtype=np.float32)
        
        # 가장자리 부드럽게
        fade_size = min(20, min(exp_x2-exp_x1, exp_y2-exp_y1) // 4)
        
        # 상하좌우 페이드
        for i in range(fade_size):
            alpha = i / fade_size
            mask[i, :] *= alpha  # 위
            mask[-(i+1), :] *= alpha  # 아래
            mask[:, i] *= alpha  # 왼쪽
            mask[:, -(i+1)] *= alpha  # 오른쪽
        
        # 블러 적용
        image[exp_y1:exp_y2, exp_x1:exp_x2] = (
            extra_blur * mask + 
            image[exp_y1:exp_y2, exp_x1:exp_x2] * (1 - mask)
        ).astype(np.uint8)
        
        return image
    
    def detect_faces(self, frame):
        """얼굴 탐지"""
        # 이미지 품질 향상
        enhanced = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # YOLO 탐지 (매우 민감하게)
        results = self.model(
            enhanced,
            conf=0.05,  # 매우 낮은 임계값
            iou=0.4,
            imgsz=640,
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
    
    def run(self):
        """실시간 얼굴 마스킹 실행"""
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 웹캠에서 프레임을 읽을 수 없습니다!")
                break
            
            # 좌우 반전 (거울 효과)
            frame = cv2.flip(frame, 1)
            
            # 얼굴 탐지
            faces = self.detect_faces(frame)
            
            # 얼굴 마스킹
            for x1, y1, x2, y2, conf in faces:
                frame = self.create_smooth_blur(frame, x1, y1, x2, y2)
                
                # 탐지 정보 표시 (선택사항)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{conf:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # FPS 계산
            fps_counter += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                start_time = time.time()
            else:
                fps = 0
            
            # 정보 표시
            info_text = f'FPS: {fps:.1f} | Faces: {len(faces)}'
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 화면 출력
            cv2.imshow('🎭 실시간 얼굴 마스킹', frame)
            
            # ESC 키로 종료
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):  # S키로 스크린샷
                cv2.imwrite(f'masked_screenshot_{int(time.time())}.jpg', frame)
                print("📸 스크린샷 저장됨!")
        
        # 정리
        self.cap.release()
        cv2.destroyAllWindows()
        print("🎬 실시간 마스킹 종료")

def main():
    """메인 함수"""
    try:
        # 실시간 얼굴 마스킹 시스템 시작
        masking_system = RealTimeFaceMasking()
        
        print("\n" + "="*50)
        print("🎭 실시간 얼굴 마스킹 시스템")
        print("="*50)
        print("📹 조작법:")
        print("   - ESC: 종료")
        print("   - S: 스크린샷 저장")
        print("="*50)
        
        masking_system.run()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("🔧 해결 방법:")
        print("   1. 웹캠이 연결되어 있는지 확인")
        print("   2. 다른 프로그램이 웹캠을 사용하고 있지 않은지 확인")
        print("   3. 관리자 권한으로 실행")

if __name__ == "__main__":
    main() 