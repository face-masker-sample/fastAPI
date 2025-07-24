# 🎭 FaceMaker - AI 기반 CCTV 관제 시스템

무인모텔 CCTV 관제를 위한 AI 통합 솔루션:
- 🎙️ **다국어 STT** (한국어/중국어/영어)
- 🎭 **얼굴 마스킹** (프라이버시 보호)
- 📱 **실시간 자막** (채팅창 스타일)

## 📦 설치 방법

### 방법 1: Conda 환경 파일 사용 (권장)

```bash
# 1. 환경 생성 및 활성화
conda env create -f environment_facemaker.yml
conda activate facemaker

# 2. FFmpeg 설치 확인
ffmpeg -version
```

### 방법 2: 수동 설치

```bash
# 1. 가상환경 생성
conda create -n facemaker python=3.10 -y
conda activate facemaker

# 2. PyTorch 설치 (CPU 버전)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 3. 필수 패키지 설치
pip install -r requirements_facemaker.txt

# 4. FFmpeg 설치
conda install ffmpeg -c conda-forge
```

## 🚀 사용 방법

### 1️⃣ STT 자막 생성

```bash
# 기본 사용법
python whisper_v3_stt.py -i cctv.mp4 -o cctv_with_subtitle.mp4

# 옵션 사용법
python whisper_v3_stt.py -i cctv.mp4 -o cctv_with_subtitle.mp4 --chunk 25
```

**출력:** 원본 음성 + 실시간 스타일 자막

### 2️⃣ 얼굴 마스킹

```bash
# 기본 사용법 (cctv_fast.mp4를 입력으로 사용)
python video_face_masking_test.py

# 옵션 사용법
python video_face_masking_test.py -i cctv_with_subtitle.mp4 -o final_output.mp4 --mask-type blur --confidence 0.15
```

**출력:** 원본 음성 + 자막 + 얼굴 마스킹

### 🔄 전체 파이프라인

```bash
# 1단계: 원본 → STT 자막
python whisper_v3_stt.py -i cctv.mp4 -o cctv_fast.mp4

# 2단계: 자막 → 마스킹
python video_face_masking_test.py -i cctv_fast.mp4 -o mask_cctv_fast.mp4

# 최종 결과: mask_cctv_fast.mp4 (음성 + 자막 + 마스킹)
```

## ⚙️ 주요 기능

### 🎙️ STT (whisper_v3_stt.py)
- **모델**: OpenAI Whisper-Large-V3
- **언어**: 한국어, 중국어, 영어 자동 감지
- **특징**: 
  - 할루시네이션 방지 강화
  - 실시간 타이밍 (말 끝난 후 0.5초 지연)
  - 채팅창 스타일 자막 (`- 안녕하세요`)

### 🎭 얼굴 마스킹 (video_face_masking_test.py)
- **모델**: YOLOv8-Face-Detection
- **마스킹**: blur, pixelate, black
- **특징**:
  - 고급 얼굴 추적 (IoU + 거리 + 크기)
  - 빠른 이동 시 연속성 보장
  - 중복 추적 제거
  - 인원 수 실시간 표시

## 📋 옵션 설명

### STT 옵션
- `--chunk`: 오디오 청크 길이 (기본: 25초)
- `-i`: 입력 비디오 파일
- `-o`: 출력 비디오 파일

### 마스킹 옵션
- `--confidence`: 검출 신뢰도 (기본: 0.15)
- `--mask-type`: 마스킹 타입 (blur/pixelate/black)
- `-i`: 입력 비디오 파일 (기본: cctv_fast.mp4)
- `-o`: 출력 비디오 파일 (기본: mask_cctv_fast.mp4)

## 🔧 문제 해결

### FFmpeg 오류
```bash
# Windows
conda install ffmpeg -c conda-forge

# 또는 시스템에서 설치
# https://ffmpeg.org/download.html
```

### CUDA 오류 (GPU 사용 시)
```bash
# CPU 버전으로 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 한글 폰트 오류
- Windows: 자동으로 맑은고딕 사용
- Linux: Noto Sans CJK 설치 필요

## 📁 프로젝트 구조

```
FaceMaker/
├── whisper_v3_stt.py           # STT 자막 생성
├── video_face_masking_test.py  # 얼굴 마스킹
├── environment_facemaker.yml   # Conda 환경
├── requirements_facemaker.txt  # pip 패키지
├── README_FaceMaker.md         # 이 파일
├── cctv.mp4                   # 입력 비디오
├── cctv_fast.mp4              # STT 결과
└── mask_cctv_fast.mp4         # 최종 결과
```

## 🎯 개발자 정보

- **프로젝트**: FaceMaker
- **목적**: 무인모텔 CCTV 관제 시스템
- **기술**: Python, PyTorch, OpenCV, Whisper, YOLOv8
- **환경**: Python 3.10, CPU/GPU 호환 