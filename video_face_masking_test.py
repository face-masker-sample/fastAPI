#!/usr/bin/env python3
"""
YOLOv8 Face Detection Video Masking Test
동영상에서 얼굴을 검출하고 마스킹 처리하는 테스트 스크립트

Requirements:
- ultralytics
- huggingface_hub
- opencv-python
- numpy
- supervision
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import time
from PIL import Image, ImageDraw, ImageFont
import platform
import subprocess
import shutil
import tempfile

class VideoFaceMasking:
    def __init__(self, confidence_threshold=0.15):
        """
        YOLOv8 얼굴 마스킹 클래스 초기화
        
        Args:
            confidence_threshold (float): 신뢰도 임계값 (기본 0.15)
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.font = self.load_korean_font()
        
        # 마스킹 연속성을 위한 얼굴 추적
        self.face_tracks = []  # 추적 중인 얼굴들
        self.track_id_counter = 0  # 추적 ID 카운터
        self.max_missing_frames = 10  # 최대 놓친 프레임 수
        self.iou_threshold = 0.3  # IoU 임계값
        self.distance_threshold = 100  # 중심점 거리 임계값 (픽셀)
        self.size_ratio_threshold = 0.5  # 크기 변화 임계값
        
        self.load_model()
    
    def load_korean_font(self):
        """한글 지원 폰트 로드"""
        try:
            if platform.system() == "Windows":
                # Windows 기본 한글 폰트들 시도
                font_paths = [
                    "C:/Windows/Fonts/malgun.ttf",  # 맑은 고딕
                    "C:/Windows/Fonts/gulim.ttc",   # 굴림
                    "C:/Windows/Fonts/batang.ttc"   # 바탕
                ]
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        return ImageFont.truetype(font_path, 30)
            elif platform.system() == "Darwin":  # macOS
                return ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", 30)
            else:  # Linux
                # Linux의 경우 Noto Sans CJK 시도
                font_paths = [
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
                ]
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        return ImageFont.truetype(font_path, 30)
        except Exception as e:
            print(f"⚠️  한글 폰트 로드 실패: {e}")
        
        # 기본 폰트 사용 (한글 깨질 수 있음)
        try:
            return ImageFont.load_default()
        except:
            return None
    
    def load_model(self):
        """HuggingFace에서 YOLOv8-Face-Detection 모델 다운로드 및 로드"""
        try:
            print("📥 YOLOv8-Face-Detection 모델 다운로드 중...")
            model_path = hf_hub_download(
                repo_id="arnabdhar/YOLOv8-Face-Detection", 
                filename="model.pt"
            )
            print(f"✅ 모델 다운로드 완료: {model_path}")
            
            # YOLOv8 모델 로드
            self.model = YOLO(model_path)
            print("✅ YOLOv8 얼굴 검출 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
    
    def calculate_iou(self, box1, box2):
        """
        두 박스 간의 IoU(Intersection over Union) 계산
        """
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        # 교집합 영역 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 영역 계산
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def calculate_center_distance(self, box1, box2):
        """
        두 박스의 중심점 간 거리 계산
        """
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        center1_x = (x1_1 + x2_1) / 2
        center1_y = (y1_1 + y2_1) / 2
        center2_x = (x1_2 + x2_2) / 2
        center2_y = (y1_2 + y2_2) / 2
        
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        return distance
    
    def calculate_size_ratio(self, box1, box2):
        """
        두 박스의 크기 비율 계산
        """
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        if area1 == 0 or area2 == 0:
            return 0
            
        ratio = min(area1, area2) / max(area1, area2)
        return ratio
    
    def calculate_similarity(self, track_box, detection_box):
        """
        종합적인 유사도 계산 (IoU + 거리 + 크기)
        """
        iou = self.calculate_iou(track_box, detection_box)
        distance = self.calculate_center_distance(track_box, detection_box)
        size_ratio = self.calculate_size_ratio(track_box, detection_box)
        
        # 거리 점수 (가까울수록 높은 점수)
        distance_score = max(0, 1 - distance / self.distance_threshold)
        
        # 종합 점수 계산 (가중 평균)
        similarity = (iou * 0.4 + distance_score * 0.4 + size_ratio * 0.2)
        
        return similarity

    def detect_faces(self, frame):
        """
        프레임에서 얼굴 검출
        
        Args:
            frame: OpenCV 이미지 프레임
            
        Returns:
            list: 검출된 얼굴 박스 리스트 [(x1, y1, x2, y2, confidence), ...]
        """
        try:
            # YOLOv8 추론
            results = self.model(frame, verbose=False)
            
            faces = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 신뢰도 확인
                        confidence = float(box.conf[0])
                        if confidence >= self.confidence_threshold:
                            # 좌표 추출 (x1, y1, x2, y2)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            faces.append((x1, y1, x2, y2, confidence))
            
            return faces
            
        except Exception as e:
            print(f"❌ 얼굴 검출 오류: {e}")
            return []
    
    def update_face_tracking(self, detected_faces):
        """
        얼굴 추적 업데이트 (연속성 개선)
        """
        # 새로운 추적 리스트
        updated_tracks = []
        used_detections = set()
        
        # 기존 추적과 새 검출 매칭 (고급 유사도 기반)
        for track in self.face_tracks:
            best_match = None
            best_similarity = 0
            best_idx = -1
            
            for i, detection in enumerate(detected_faces):
                if i in used_detections:
                    continue
                    
                # 종합적인 유사도 계산
                similarity = self.calculate_similarity(track['box'], detection)
                
                # 최소 임계값 조건들
                distance = self.calculate_center_distance(track['box'], detection)
                size_ratio = self.calculate_size_ratio(track['box'], detection)
                
                # 더 관대한 조건으로 빠른 움직임 허용
                if (similarity > 0.2 and  # 전체 유사도
                    distance < self.distance_threshold and  # 거리 제한
                    size_ratio > self.size_ratio_threshold and  # 크기 변화 제한
                    similarity > best_similarity):
                    best_similarity = similarity
                    best_match = detection
                    best_idx = i
            
            if best_match is not None:
                # 기존 추적 업데이트
                track['box'] = best_match
                track['missing_count'] = 0
                track['confidence'] = best_match[4]
                updated_tracks.append(track)
                used_detections.add(best_idx)
            else:
                # 매칭되지 않은 추적 - 놓친 횟수 증가
                track['missing_count'] += 1
                if track['missing_count'] < self.max_missing_frames:
                    # 이전 위치 유지 (연속성)
                    updated_tracks.append(track)
        
        # 새로운 검출 추가
        for i, detection in enumerate(detected_faces):
            if i not in used_detections:
                new_track = {
                    'id': self.track_id_counter,
                    'box': detection,
                    'missing_count': 0,
                    'confidence': detection[4]
                }
                updated_tracks.append(new_track)
                self.track_id_counter += 1
        
        # 중복된 추적 제거 (같은 사람을 여러 번 추적하는 것 방지)
        final_tracks = self.remove_duplicate_tracks(updated_tracks)
        self.face_tracks = final_tracks
        
        # 추적된 얼굴 박스들 반환
        return [track['box'] for track in self.face_tracks]
    
    def remove_duplicate_tracks(self, tracks):
        """
        중복된 추적 제거 (같은 사람을 여러 개로 인식하는 것 방지)
        """
        if len(tracks) <= 1:
            return tracks
            
        final_tracks = []
        
        for i, track1 in enumerate(tracks):
            is_duplicate = False
            
            for j, track2 in enumerate(final_tracks):
                # 이미 추가된 추적과 비교
                similarity = self.calculate_similarity(track1['box'], track2['box'])
                distance = self.calculate_center_distance(track1['box'], track2['box'])
                
                # 너무 비슷하면 중복으로 판단
                if similarity > 0.7 or distance < 50:
                    is_duplicate = True
                    # 더 신뢰도 높은 것으로 유지
                    if track1['confidence'] > track2['confidence']:
                        final_tracks[j] = track1
                    break
            
            if not is_duplicate:
                final_tracks.append(track1)
        
        return final_tracks
    

    
    def add_person_count_text(self, frame, person_count):
        """
        프레임 좌측 상단에 한글로 인원 수 표시
        
        Args:
            frame: OpenCV 이미지 프레임
            person_count: 검출된 인원 수
            
        Returns:
            numpy.ndarray: 텍스트가 추가된 프레임
        """
        try:
            # OpenCV 프레임을 PIL Image로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # 텍스트 내용
            text = f"인원: {person_count}명"
            
            # 텍스트 위치 (좌측 상단)
            text_position = (20, 20)
            
            # 텍스트 배경을 위한 반투명 박스
            if self.font:
                # 텍스트 크기 계산
                bbox = draw.textbbox(text_position, text, font=self.font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # 폰트가 없을 경우 대략적인 크기
                text_width = len(text) * 15
                text_height = 25
            
            # 배경 박스 그리기 (반투명 검은색)
            background_box = [
                text_position[0] - 10,
                text_position[1] - 5,
                text_position[0] + text_width + 10,
                text_position[1] + text_height + 5
            ]
            
            # 반투명 배경
            overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(background_box, fill=(0, 0, 0, 128))  # 반투명 검은색
            
            # 배경과 원본 이미지 합성
            pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(pil_image)
            
            # 텍스트 그리기 (흰색)
            if self.font:
                draw.text(text_position, text, font=self.font, fill=(255, 255, 255))
            else:
                # 폰트가 없을 경우 기본 폰트 (영어/숫자만 표시)
                draw.text(text_position, f"Count: {person_count}", fill=(255, 255, 255))
            
            # PIL Image를 다시 OpenCV 프레임으로 변환
            frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return frame_with_text
            
        except Exception as e:
            print(f"⚠️  텍스트 추가 실패: {e}")
            # 실패 시 OpenCV로 기본 텍스트 추가
            cv2.putText(frame, f"Count: {person_count}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            return frame

    def apply_face_masking(self, frame, faces, mask_type="blur"):
        """
        검출된 얼굴에 마스킹 적용
        
        Args:
            frame: OpenCV 이미지 프레임
            faces: 검출된 얼굴 박스 리스트
            mask_type: 마스킹 타입 ("blur", "pixelate", "black")
            
        Returns:
            numpy.ndarray: 마스킹된 프레임
        """
        masked_frame = frame.copy()
        
        for face in faces:
            x1, y1, x2, y2, confidence = face
            
            # 얼굴 영역 추출
            face_region = masked_frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                continue
                
            # 마스킹 타입별 처리
            if mask_type == "blur":
                # 가우시안 블러 적용 (강도 높임)
                blurred = cv2.GaussianBlur(face_region, (51, 51), 30)
                masked_frame[y1:y2, x1:x2] = blurred
                
            elif mask_type == "pixelate":
                # 픽셀화 효과
                h, w = face_region.shape[:2]
                temp = cv2.resize(face_region, (w//10, h//10), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                masked_frame[y1:y2, x1:x2] = pixelated
                
            elif mask_type == "black":
                # 검은색 박스로 마스킹
                masked_frame[y1:y2, x1:x2] = 0
            
            # 디버깅용: 신뢰도 표시 (선택사항)
            # cv2.putText(masked_frame, f'{confidence:.2f}', (x1, y1-10), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return masked_frame
    
    def process_video(self, input_path, output_path, mask_type="blur", show_progress=True):
        """
        동영상 파일 전체 처리
        
        Args:
            input_path (str): 입력 동영상 경로
            output_path (str): 출력 동영상 경로
            mask_type (str): 마스킹 타입
            show_progress (bool): 진행률 표시 여부
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
        
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(input_path)
        
        # 비디오 정보 가져오기
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 입력 동영상 정보:")
        print(f"   - 해상도: {width}x{height}")
        print(f"   - FPS: {fps}")
        print(f"   - 총 프레임: {total_frames}")
        print(f"   - 신뢰도 임계값: {self.confidence_threshold}")
        
        # 비디오 라이터 객체 생성 (임시 파일, 음성 없음)
        temp_video = tempfile.mktemp(suffix='_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print(f"🎬 동영상 처리 시작...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 얼굴 검출
                detected_faces = self.detect_faces(frame)
                
                # 얼굴 추적으로 연속성 개선
                tracked_faces = self.update_face_tracking(detected_faces)
                
                # 마스킹 적용
                masked_frame = self.apply_face_masking(frame, tracked_faces, mask_type)
                
                # 인원 수 텍스트 추가
                final_frame = self.add_person_count_text(masked_frame, len(tracked_faces))
                
                # 프레임 저장
                out.write(final_frame)
                
                frame_count += 1
                
                # 진행률 표시
                if show_progress and frame_count % 30 == 0:  # 30프레임마다 표시
                    progress = (frame_count / total_frames) * 100
                    elapsed_time = time.time() - start_time
                    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    print(f"   진행률: {progress:.1f}% ({frame_count}/{total_frames}) "
                          f"- 처리 속도: {avg_fps:.1f} FPS - 현재 인원: {len(tracked_faces)}명 "
                          f"(검출: {len(detected_faces)}, 추적: {len(tracked_faces)})")
        
        finally:
            # 리소스 정리
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        # 원본 음성과 마스킹된 비디오 합치기
        print("🔊 원본 음성과 마스킹된 비디오 합치는 중...")
        
        merge_command = [
            'ffmpeg', '-i', temp_video,  # 마스킹된 비디오 (음성 없음)
            '-i', input_path,  # 원본 비디오 (음성 포함)
            '-c:v', 'copy',  # 비디오 코덱 복사 (빠른 처리)
            '-c:a', 'aac',   # 오디오 코덱 AAC
            '-map', '0:v:0',  # 첫 번째 입력의 비디오 스트림
            '-map', '1:a:0',  # 두 번째 입력의 오디오 스트림
            '-shortest',  # 짧은 쪽에 맞춤
            '-y',  # 덮어쓰기
            output_path
        ]
        
        try:
            subprocess.run(merge_command, capture_output=True, text=True, check=True)
            print("✅ 음성과 비디오 합치기 완료!")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ 음성 합치기 실패, 대체 방법 시도...")
            print(f"오류: {e.stderr}")
            
            # 대체 방법: 더 호환성 높은 명령어
            fallback_command = [
                'ffmpeg', '-i', temp_video,
                '-i', input_path,
                '-c:v', 'libx264',  # H.264 코덱 사용
                '-c:a', 'aac',
                '-map', '0:v',  # 마스킹된 비디오의 영상
                '-map', '1:a',  # 원본의 음성
                '-shortest',
                '-y',
                output_path
            ]
            
            try:
                subprocess.run(fallback_command, capture_output=True, text=True, check=True)
                print("✅ 대체 방법으로 음성과 비디오 합치기 완료!")
            except subprocess.CalledProcessError as e2:
                print(f"❌ 음성 합치기 최종 실패: {e2.stderr}")
                # 마스킹만 있는 비디오라도 저장
                shutil.move(temp_video, output_path)
                print(f"⚠️ 음성 없이 마스킹만 저장됨: {output_path}")
        
        # 임시 파일 정리
        try:
            os.unlink(temp_video)
        except:
            pass
        
        processing_time = time.time() - start_time
        print(f"✅ 동영상 처리 완료!")
        print(f"   - 총 처리 시간: {processing_time:.2f}초")
        print(f"   - 평균 처리 속도: {frame_count/processing_time:.2f} FPS")
        print(f"   - 출력 파일: {output_path} (음성 포함)")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='YOLOv8 얼굴 마스킹 동영상 처리')
    parser.add_argument('--input', '-i', type=str, default='cctv_fast.mp4',
                       help='입력 동영상 파일 경로 (기본: cctv_fast.mp4)')
    parser.add_argument('--output', '-o', type=str, default='mask_cctv_fast.mp4',
                       help='출력 동영상 파일 경로 (기본: mask_cctv_fast.mp4)')
    parser.add_argument('--confidence', '-c', type=float, default=0.15,
                       help='신뢰도 임계값 (기본: 0.15)')
    parser.add_argument('--mask-type', '-m', type=str, default='blur',
                       choices=['blur', 'pixelate', 'black'],
                       help='마스킹 타입 (기본: blur)')
    
    args = parser.parse_args()
    
    try:
        # 얼굴 마스킹 객체 생성
        face_masker = VideoFaceMasking(confidence_threshold=args.confidence)
        
        # 동영상 처리
        face_masker.process_video(
            input_path=args.input,
            output_path=args.output,
            mask_type=args.mask_type
        )
        
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 