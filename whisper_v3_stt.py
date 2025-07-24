#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceMaker STT with Whisper-Large-V3
HuggingFace 공식 whisper-large-v3 모델 사용
"""

import cv2
import torch
import numpy as np
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings("ignore")

class WhisperV3STT:
    def __init__(self):
        """
        Whisper-Large-V3 기반 다국어 STT 클래스
        지원 언어: 한국어, 중국어, 영어 자동 감지
        """
        print("🚀 Whisper-Large-V3 다국어 모델 로딩 중...")
        
        # GPU 사용 가능 여부 확인
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"💻 사용 디바이스: {self.device}")
        print(f"🔢 데이터 타입: {self.torch_dtype}")
        
        # 모델 ID
        self.model_id = "openai/whisper-large-v3"
        
        # 모델 로드
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        
        # 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # 파이프라인 생성
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
        print("✅ Whisper-Large-V3 모델 로딩 완료!")
    
    def extract_audio_with_ffmpeg(self, video_path, output_path=None):
        """
        FFmpeg를 사용하여 비디오에서 오디오 추출
        """
        if output_path is None:
            output_path = video_path.replace('.mp4', '_audio.wav')
        
        print(f"🎵 오디오 추출 중: {video_path} -> {output_path}")
        
        command = [
            'ffmpeg', '-i', video_path,
            '-vn',  # 비디오 스트림 무시
            '-acodec', 'pcm_s16le',  # 16비트 PCM
            '-ar', '16000',  # 16kHz 샘플링
            '-ac', '1',  # 모노 채널
            '-y',  # 덮어쓰기
            output_path
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("✅ 오디오 추출 완료!")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ 오디오 추출 실패: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return None
    
    def split_audio_chunks(self, audio_path, chunk_duration=30):
        """
        오디오를 청크로 분할 (whisper의 최대 입력 길이: 30초)
        """
        print(f"✂️ 오디오 청크 분할 중... (청크 길이: {chunk_duration}초)")
        
        # 오디오 길이 확인
        command = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', audio_path
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            print(f"📏 총 오디오 길이: {duration:.2f}초")
        except:
            print("⚠️ 오디오 길이 확인 실패, 기본값 사용")
            duration = 300  # 5분으로 가정
        
        # 청크 분할
        chunks = []
        temp_dir = tempfile.mkdtemp()
        
        for i, start_time in enumerate(range(0, int(duration), chunk_duration)):
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
            
            command = [
                'ffmpeg', '-i', audio_path,
                '-ss', str(start_time),
                '-t', str(chunk_duration),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y', chunk_path
            ]
            
            try:
                subprocess.run(command, capture_output=True, check=True)
                chunks.append((chunk_path, start_time, min(start_time + chunk_duration, duration)))
                print(f"📦 청크 {i+1} 생성: {start_time}~{min(start_time + chunk_duration, duration)}초")
            except subprocess.CalledProcessError:
                print(f"❌ 청크 {i+1} 생성 실패")
        
        return chunks
    
    def transcribe_chunk(self, audio_path, chunk_start_time):
        """
        단일 오디오 청크를 전사 (문장별 타임스탬프 포함)
        """
        try:
            # 문장별 타임스탬프 설정 (할루시네이션 방지 강화)
            generate_kwargs = {
                "max_new_tokens": 224,
                "num_beams": 1,
                "condition_on_prev_tokens": False,
                "compression_ratio_threshold": 1.35,
                "temperature": (0.0, 0.2, 0.4),  # 3단계로 줄여서 속도↑, 정확도 유지
                "logprob_threshold": -0.8,  # -1.0 -> -0.8로 더 엄격하게
                "no_speech_threshold": 0.6,  # 0.5 -> 0.6으로 더 엄격하게
                "task": "transcribe"
            }
            
            # 문장별 타임스탬프 얻기
            result = self.pipe(audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)
            
            if result and 'chunks' in result:
                # 문장별로 분리된 결과 처리
                sentences = []
                for chunk in result['chunks']:
                    text = chunk['text'].strip()
                    if len(text) > 3 and not self._is_hallucination(text):
                        # 문장 종료 시점으로 타임스탬프 설정 (실시간 느낌)
                        start_time = chunk_start_time + chunk['timestamp'][0]
                        end_time = chunk_start_time + chunk['timestamp'][1]
                        
                        # 자막 표시 지연: 말이 끝나는 시점 + 0.5초 후에 표시
                        display_time = end_time + 0.5
                        
                        sentences.append({
                            'timestamp': display_time,
                            'text': text,
                            'speech_start': start_time,
                            'speech_end': end_time
                        })
                return sentences
            
            return []
            
        except Exception as e:
            print(f"⚠️ 전사 중 오류: {e}")
            return None
    
    def _is_hallucination(self, text):
        """
        환각 텍스트 감지 (다국어 패턴 매칭)
        """
        # 텍스트 길이 체크 (너무 짧거나 너무 길면 의심)
        if len(text.strip()) < 2 or len(text.strip()) > 200:
            return True
            
        # 한국어/중국어/영어 환각 패턴
        hallucination_patterns = [
            # 한국어 환각 (더 많이 추가)
            "이제는", "지금은", "그렇다면", "그런데", "하지만", "그래서", "그리고",
            "신이", "약간", "되나", "이겼", "일어날", "수 있어", "그래",
            "어디", "뭐야", "아니", "맞아", "그냥", "근데", "진짜",
            # 영어 환각
            "thanks for watching", "subscribe", "like", "comment", "share",
            "please subscribe", "hit the bell", "notification",
            "you know", "i think", "right now", "okay okay",
            # 중국어 환각
            "谢谢观看", "请订阅", "点赞", "评论", "分享", "感谢收看"
        ]
        
        text_lower = text.lower().strip()
        
        # 패턴 매칭
        for pattern in hallucination_patterns:
            if pattern in text_lower:
                return True
        
        # 반복되는 단어 검사 (같은 단어가 여러번)
        words = text.split()
        if len(words) > 1:
            unique_words = set(words)
            if len(unique_words) < len(words) * 0.5:  # 50% 이상 중복
                return True
        
        # 의미없는 짧은 문장 검사
        meaningless_shorts = ["아", "어", "음", "오", "네", "예", "응", "ok", "okay"]
        if text_lower in meaningless_shorts:
            return True
            
        # 반복 패턴 검사
        if len(words) > 2:
            if len(set(words)) < len(words) * 0.5:  # 50% 이상 중복
                return True
        
        return False
    
    def get_korean_font(self, size=24):
        """
        한국어 폰트 로드
        """
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",  # Windows 맑은고딕
            "C:/Windows/Fonts/gulim.ttc",   # Windows 굴림
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except:
                    continue
        
        # 기본 폰트 사용
        try:
            return ImageFont.load_default()
        except:
            return None
    
    def add_subtitle_to_frame(self, frame, subtitle_lines, max_lines=10):
        """
        프레임 우측에 채팅창 스타일 자막 추가
        """
        height, width = frame.shape[:2]
        
        # PIL 이미지로 변환
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        # 폰트 설정
        font = self.get_korean_font(size=20)
        
        # 우측 영역 설정 (크기 축소)
        subtitle_width = width // 6  # 1/3 -> 1/6으로 축소
        subtitle_x = width - subtitle_width - 10
        
        # 배경 영역 그리기 (투명도 증가, 높이 더 크게)
        bg_height = int(height * 0.8)  # 높이를 80%로 더 크게
        bg_rect = [subtitle_x - 5, 10, width - 5, 10 + bg_height]
        
        # 반투명 검은 배경 (투명도 높임)
        overlay = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(bg_rect, fill=(0, 0, 0, 100), outline=(255, 255, 255, 150))
        frame_pil = Image.alpha_composite(frame_pil.convert('RGBA'), overlay).convert('RGB')
        
        # 배경 적용 후 다시 draw 객체 생성
        draw = ImageDraw.Draw(frame_pil)
        
        # 제목 (원래 크기로)
        title_text = "🌍 다국어 자막"
        title_font = self.get_korean_font(size=20)  # 원래 크기
        if title_font:
            draw.text((subtitle_x, 20), title_text, fill=(255, 255, 0), font=title_font)
        else:
            draw.text((subtitle_x, 20), title_text, fill=(255, 255, 0))
        
        # 자막 표시 (최근 max_lines개만)
        max_lines_adjusted = min(max_lines, (bg_height - 80) // 30)  # 원래 줄간격으로 조정
        recent_lines = subtitle_lines[-max_lines_adjusted:] if len(subtitle_lines) > max_lines_adjusted else subtitle_lines
        
        y_offset = 60  # 원래 위치
        line_height = 30  # 원래 줄 간격
        
        # 원래 폰트 크기로 자막 표시 (앞에 - 추가)
        subtitle_font = self.get_korean_font(size=20)  # 원래 크기
        for i, line in enumerate(recent_lines):
            display_line = f"- {line}"  # 앞에 - 추가
            if subtitle_font:
                draw.text((subtitle_x, y_offset + i * line_height), display_line, fill=(255, 255, 255), font=subtitle_font)
            else:
                draw.text((subtitle_x, y_offset + i * line_height), display_line, fill=(255, 255, 255))
        
        # OpenCV 형식으로 다시 변환
        frame_with_subtitle = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        return frame_with_subtitle
    
    def process_video(self, video_path, output_path, chunk_duration=25):
        """
        비디오에 STT 자막 추가
        """
        print(f"🎬 비디오 처리 시작: {video_path}")
        
        # 1. 오디오 추출
        temp_audio = tempfile.mktemp(suffix='.wav')
        audio_path = self.extract_audio_with_ffmpeg(video_path, temp_audio)
        if not audio_path:
            return False
        
        # 2. 오디오 청크 분할
        chunks = self.split_audio_chunks(audio_path, chunk_duration)
        if not chunks:
            print("❌ 오디오 청크 분할 실패")
            return False
        
        # 3. STT 처리 (문장별 타임스탬프)
        print("🎙️ 음성 인식 중 (문장별 분리)...")
        all_sentences = []
        
        for i, (chunk_path, start_time, end_time) in enumerate(chunks):
            print(f"📝 청크 {i+1}/{len(chunks)} 처리 중... ({start_time:.1f}~{end_time:.1f}초)")
            sentences = self.transcribe_chunk(chunk_path, start_time)
            
            if sentences:
                all_sentences.extend(sentences)
                for sentence in sentences:
                    print(f"✅ 음성: [{sentence['speech_start']:.1f}~{sentence['speech_end']:.1f}s] → 자막: [{sentence['timestamp']:.1f}s] {sentence['text']}")
            else:
                print("⚪ 음성 없음")
        
        # 시간순으로 정렬
        all_sentences.sort(key=lambda x: x['timestamp'])
        
        # 4. 비디오에 자막 추가 (실시간 스타일)
        print("🎥 비디오에 실시간 스타일 자막 추가 중...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 비디오 라이터 설정 (음성 없이 임시 파일)
        temp_video = tempfile.mktemp(suffix='_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        subtitle_lines = []
        added_sentences = set()  # 이미 추가된 문장 추적
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 현재 시간 계산
            current_time = frame_count / fps
            
            # 아직 추가되지 않은 문장 중에서 시간이 지난 것들 추가
            for sentence in all_sentences:
                sentence_id = f"{sentence['timestamp']:.1f}_{sentence['text']}"
                
                # 해당 문장의 시작 시간이 지났고, 아직 추가되지 않았다면
                if current_time >= sentence['timestamp'] and sentence_id not in added_sentences:
                    subtitle_lines.append(f"{sentence['text']}")
                    added_sentences.add(sentence_id)
                    print(f"📺 [{current_time:.1f}s] 자막 표시: {sentence['text']} (음성: {sentence['speech_start']:.1f}~{sentence['speech_end']:.1f}s)")
            
            # 자막이 포함된 프레임 생성
            frame_with_subtitle = self.add_subtitle_to_frame(frame, subtitle_lines)
            out.write(frame_with_subtitle)
            
            frame_count += 1
            
            # 진행률 표시
            if frame_count % (fps * 10) == 0:  # 10초마다
                progress = (frame_count / total_frames) * 100
                print(f"⏳ 진행률: {progress:.1f}%")
        
        cap.release()
        out.release()
        
        # 5. 원본 음성과 자막 비디오 합치기
        print("🔊 원본 음성과 자막 비디오 합치는 중...")
        merge_command = [
            'ffmpeg', '-i', temp_video,  # 자막이 포함된 비디오 (음성 없음)
            '-i', video_path,  # 원본 비디오 (음성 포함)
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
            print("✅ 음성과 자막 합치기 완료!")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ 음성 합치기 실패, 대체 방법 시도...")
            print(f"오류: {e.stderr}")
            
            # 대체 방법: 더 호환성 높은 명령어
            fallback_command = [
                'ffmpeg', '-i', temp_video,
                '-i', video_path,
                '-c:v', 'libx264',  # H.264 코덱 사용
                '-c:a', 'aac',
                '-map', '0:v',  # 자막 비디오의 영상
                '-map', '1:a',  # 원본의 음성
                '-shortest',
                '-y',
                output_path
            ]
            
            try:
                subprocess.run(fallback_command, capture_output=True, text=True, check=True)
                print("✅ 대체 방법으로 음성과 자막 합치기 완료!")
            except subprocess.CalledProcessError as e2:
                print(f"❌ 음성 합치기 최종 실패: {e2.stderr}")
                # 자막만 있는 비디오라도 저장
                import shutil
                shutil.move(temp_video, output_path)
                print(f"⚠️ 음성 없이 자막만 저장됨: {output_path}")
        
        # 임시 파일 정리
        try:
            os.unlink(temp_audio)
            os.unlink(temp_video)
            for chunk_path, _, _ in chunks:
                os.unlink(chunk_path)
        except:
            pass
        
        print(f"🎉 완료! 음성과 자막이 포함된 파일: {output_path}")
        return True

def main():
    """
    메인 함수
    """
    parser = argparse.ArgumentParser(
        description="FaceMaker STT with Whisper-Large-V3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python whisper_v3_stt.py -i cctv.mp4 -o cctv_with_subtitle.mp4
  python whisper_v3_stt.py -i cctv.mp4 -o cctv_with_subtitle.mp4 --chunk 20
        """
    )
    
    parser.add_argument(
        '-i', '--input', 
        required=True, 
        help='입력 비디오 파일 경로'
    )
    
    parser.add_argument(
        '-o', '--output', 
        required=True, 
        help='출력 비디오 파일 경로'
    )
    
    parser.add_argument(
        '--chunk', 
        type=int, 
        default=25, 
        help='오디오 청크 길이 (초, 기본값: 25 - 정확도와 속도 균형)'
    )
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        return 1
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    print("=" * 60)
    print("🎯 FaceMaker STT with Whisper-Large-V3")
    print(f"📁 입력: {args.input}")
    print(f"💾 출력: {args.output}")
    print(f"⏱️ 청크 길이: {args.chunk}초")
    print("=" * 60)
    
    try:
        stt = WhisperV3STT()
        success = stt.process_video(args.input, args.output, args.chunk)
        
        if success:
            print("\n🎉 모든 작업이 완료되었습니다!")
            return 0
        else:
            print("\n❌ 작업 중 오류가 발생했습니다.")
            return 1
            
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 