"""
실시간 웹캠 얼굴 마스킹 프로그램 - 이모지 버전 😊
YOLOv8-Face 모델을 사용하여 실시간으로 얼굴을 탐지하고 노란색 스마일 이모지로 대체
"""

import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os
import time

class RealTimeFaceEmojiMasking:
    def __init__(self):
        """실시간 얼굴 이모지 마스킹 시스템 초기화"""
        print("😊 실시간 얼굴 이모지 마스킹 시스템 초기화 중...")
        
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
    
    def create_emoji_overlay(self, image, x1, y1, x2, y2, emoji_type="smile"):
        """노란색 이모지로 얼굴 대체"""
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
        
        if emoji_type == "smile":
            self.draw_smile_emoji(image, center_x, center_y, emoji_radius)
        elif emoji_type == "laugh":
            self.draw_laugh_emoji(image, center_x, center_y, emoji_radius)
        elif emoji_type == "wink":
            self.draw_wink_emoji(image, center_x, center_y, emoji_radius)
        
        return image
    
    def draw_smile_emoji(self, image, center_x, center_y, radius):
        """스마일 이모지 그리기 😊"""
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
    
    def draw_laugh_emoji(self, image, center_x, center_y, radius):
        """웃음 이모지 그리기 😄"""
        # 노란색 배경 원
        cv2.circle(image, (center_x, center_y), radius, (0, 215, 255), -1)
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 0), 3)
        
        # 웃는 눈 (작은 호)
        eye_offset_x = radius // 3
        eye_offset_y = radius // 4
        eye_width = radius // 6
        eye_height = radius // 12
        
        # 왼쪽 웃는 눈
        cv2.ellipse(image, (center_x - eye_offset_x, center_y - eye_offset_y), (eye_width, eye_height), 0, 0, 180, (0, 0, 0), 3)
        
        # 오른쪽 웃는 눈
        cv2.ellipse(image, (center_x + eye_offset_x, center_y - eye_offset_y), (eye_width, eye_height), 0, 0, 180, (0, 0, 0), 3)
        
        # 큰 웃는 입
        mouth_y = center_y + radius // 6
        mouth_width = int(radius // 1.5)
        mouth_height = int(radius // 2)
        
        cv2.ellipse(image, (center_x, mouth_y), (mouth_width, mouth_height), 0, 0, 180, (0, 0, 0), 4)
        
        # 입 안쪽 (빨간색)
        if mouth_width > 5 and mouth_height > 5:
            cv2.ellipse(image, (center_x, mouth_y), (mouth_width-5, mouth_height-5), 0, 0, 180, (0, 0, 200), -1)
    
    def draw_wink_emoji(self, image, center_x, center_y, radius):
        """윙크 이모지 그리기 😉"""
        # 노란색 배경 원
        cv2.circle(image, (center_x, center_y), radius, (0, 215, 255), -1)
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 0), 3)
        
        # 눈
        eye_offset_x = radius // 3
        eye_offset_y = radius // 4
        eye_radius = radius // 8
        
        # 왼쪽 눈 (윙크 - 작은 호)
        eye_width = radius // 6
        eye_height = radius // 12
        cv2.ellipse(image, (center_x - eye_offset_x, center_y - eye_offset_y), (eye_width, eye_height), 0, 0, 180, (0, 0, 0), 3)
        
        # 오른쪽 눈 (일반)
        cv2.circle(image, (center_x + eye_offset_x, center_y - eye_offset_y), eye_radius, (0, 0, 0), -1)
        
        # 입 그리기
        mouth_y = center_y + radius // 4
        mouth_width = int(radius // 2.5)
        mouth_height = int(radius // 4)
        
        cv2.ellipse(image, (center_x, mouth_y), (mouth_width, mouth_height), 0, 0, 180, (0, 0, 0), 4)
    
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
        """실시간 얼굴 이모지 마스킹 실행"""
        fps_counter = 0
        start_time = time.time()
        emoji_types = ["smile", "laugh", "wink"]
        emoji_index = 0  # 기본값: 스마일 이모지
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 웹캠에서 프레임을 읽을 수 없습니다!")
                break
            
            # 좌우 반전 (거울 효과)
            frame = cv2.flip(frame, 1)
            
            # 얼굴 탐지
            faces = self.detect_faces(frame)
            
            # 얼굴 이모지 마스킹 (기본: 스마일)
            current_emoji = emoji_types[emoji_index]
            for x1, y1, x2, y2, conf in faces:
                frame = self.create_emoji_overlay(frame, x1, y1, x2, y2, current_emoji)
                
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
            info_text = f'FPS: {fps:.1f} | Faces: {len(faces)} | Emoji: {current_emoji}'
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 화면 출력
            cv2.imshow('😊 실시간 얼굴 이모지 마스킹', frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('1'):  # 1키: 스마일
                emoji_index = 0
                print("😊 스마일 이모지 선택")
            elif key == ord('2'):  # 2키: 웃음
                emoji_index = 1
                print("😄 웃음 이모지 선택")
            elif key == ord('3'):  # 3키: 윙크
                emoji_index = 2
                print("😉 윙크 이모지 선택")
            elif key == ord('s'):  # S키로 스크린샷
                cv2.imwrite(f'emoji_screenshot_{int(time.time())}.jpg', frame)
                print("📸 이모지 스크린샷 저장됨!")
        
        # 정리
        self.cap.release()
        cv2.destroyAllWindows()
        print("🎬 실시간 이모지 마스킹 종료")

def main():
    """메인 함수"""
    try:
        # 실시간 얼굴 이모지 마스킹 시스템 시작
        emoji_system = RealTimeFaceEmojiMasking()
        
        print("\n" + "="*60)
        print("😊 실시간 얼굴 스마일 마스킹 시스템")
        print("="*60)
        print("📹 조작법:")
        print("   - ESC: 종료")
        print("   - 1: 스마일 이모지 😊 (기본)")
        print("   - 2: 웃음 이모지 😄") 
        print("   - 3: 윙크 이모지 😉")
        print("   - S: 스크린샷 저장")
        print("   - 기본: 항상 스마일 이모지로 마스킹")
        print("="*60)
        
        emoji_system.run()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("🔧 해결 방법:")
        print("   1. 웹캠이 연결되어 있는지 확인")
        print("   2. 다른 프로그램이 웹캠을 사용하고 있지 않은지 확인")
        print("   3. 관리자 권한으로 실행")

if __name__ == "__main__":
    main() 