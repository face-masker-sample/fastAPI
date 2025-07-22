// CCTV 관제 대시보드 JavaScript
class CCTVDashboard {
    constructor() {
        this.streamWs = null;
        this.controlWs = null;
        this.isRecording = false;
        this.stats = {
            fps: 0,
            faces: 0,
            totalFaces: 0,
            uptime: 0
        };
        
        this.init();
    }
    
    init() {
        console.log('🚀 CCTV Dashboard 초기화');
        this.setupWebSockets();
        this.setupEventListeners();
        this.updateStats();
        
        // 주기적 업데이트
        setInterval(() => this.updateUptime(), 1000);
    }
    
    setupWebSockets() {
        // 실시간 스트림 WebSocket
        this.connectStreamWebSocket();
        
        // 제어 WebSocket
        this.connectControlWebSocket();
    }
    
    connectStreamWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
        
        this.streamWs = new WebSocket(wsUrl);
        
        this.streamWs.onopen = () => {
            console.log('🔌 스트림 WebSocket 연결됨');
            this.updateConnectionStatus(true);
        };
        
        this.streamWs.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleStreamData(data);
            } catch (error) {
                console.error('❌ 스트림 데이터 파싱 오류:', error);
            }
        };
        
        this.streamWs.onclose = () => {
            console.log('🔌 스트림 WebSocket 연결 해제');
            this.updateConnectionStatus(false);
            
            // 재연결 시도
            setTimeout(() => this.connectStreamWebSocket(), 3000);
        };
        
        this.streamWs.onerror = (error) => {
            console.error('❌ 스트림 WebSocket 오류:', error);
        };
    }
    
    connectControlWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/control`;
        
        this.controlWs = new WebSocket(wsUrl);
        
        this.controlWs.onopen = () => {
            console.log('🔌 제어 WebSocket 연결됨');
        };
        
        this.controlWs.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleControlResponse(data);
            } catch (error) {
                console.error('❌ 제어 응답 파싱 오류:', error);
            }
        };
        
        this.controlWs.onclose = () => {
            console.log('🔌 제어 WebSocket 연결 해제');
            
            // 재연결 시도
            setTimeout(() => this.connectControlWebSocket(), 3000);
        };
    }
    
    handleStreamData(data) {
        // 비디오 프레임 업데이트 (성능 최적화)
        if (data.frame) {
            const videoElement = document.getElementById('video-stream');
            if (videoElement) {
                // 이전 src 해제로 메모리 최적화
                if (videoElement.src.startsWith('data:')) {
                    URL.revokeObjectURL(videoElement.src);
                }
                videoElement.src = `data:image/jpeg;base64,${data.frame}`;
            }
        }
        
        // 통계 업데이트
        this.stats.fps = data.fps || 0;
        this.stats.faces = data.faces_count || 0;
        this.stats.totalFaces += data.faces_count || 0;
        this.isRecording = data.recording || false;
        
        this.updateStats();
    }
    
    handleControlResponse(data) {
        console.log('제어 응답:', data);
        
        if (data.command === 'start_recording') {
            this.isRecording = data.success;
            this.updateRecordingStatus();
        } else if (data.command === 'stop_recording') {
            this.isRecording = !data.success;
            this.updateRecordingStatus();
        } else if (data.command === 'take_screenshot') {
            if (data.success) {
                this.showNotification(`스크린샷 저장됨: ${data.filename}`, 'success');
            } else {
                this.showNotification('스크린샷 실패', 'error');
            }
        }
    }
    
    setupEventListeners() {
        // 녹화 시작 버튼
        document.getElementById('start-recording')?.addEventListener('click', () => {
            this.sendControlCommand('start_recording');
        });
        
        // 녹화 중지 버튼
        document.getElementById('stop-recording')?.addEventListener('click', () => {
            this.sendControlCommand('stop_recording');
        });
        
        // 스크린샷 버튼
        document.getElementById('take-screenshot')?.addEventListener('click', () => {
            this.sendControlCommand('take_screenshot');
        });
        
        // 통계 새로고침 버튼
        document.getElementById('refresh-stats')?.addEventListener('click', () => {
            this.sendControlCommand('get_stats');
        });
    }
    
    sendControlCommand(command, data = {}) {
        if (this.controlWs && this.controlWs.readyState === WebSocket.OPEN) {
            const message = {
                command: command,
                ...data
            };
            
            this.controlWs.send(JSON.stringify(message));
            console.log('제어 명령 전송:', message);
        } else {
            console.error('❌ 제어 WebSocket이 연결되지 않음');
            this.showNotification('제어 시스템에 연결되지 않음', 'error');
        }
    }
    
    updateStats() {
        // FPS 업데이트
        const fpsElement = document.getElementById('fps-value');
        if (fpsElement) {
            fpsElement.textContent = this.stats.fps.toFixed(1);
        }
        
        // 탐지된 얼굴 수 업데이트
        const facesElement = document.getElementById('faces-value');
        if (facesElement) {
            facesElement.textContent = this.stats.faces;
        }
        
        // 총 탐지된 얼굴 수 업데이트
        const totalFacesElement = document.getElementById('total-faces-value');
        if (totalFacesElement) {
            totalFacesElement.textContent = this.stats.totalFaces;
        }
        
        // 녹화 상태 업데이트
        this.updateRecordingStatus();
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.className = connected ? 'status-indicator status-online' : 'status-indicator status-offline';
        }
        
        const statusTextElement = document.getElementById('connection-text');
        if (statusTextElement) {
            statusTextElement.textContent = connected ? '온라인' : '오프라인';
        }
        
        // 비디오 플레이스홀더 표시/숨기기
        const videoElement = document.getElementById('video-stream');
        const placeholderElement = document.getElementById('video-placeholder');
        
        if (connected) {
            if (videoElement) videoElement.style.display = 'block';
            if (placeholderElement) placeholderElement.style.display = 'none';
        } else {
            if (videoElement) videoElement.style.display = 'none';
            if (placeholderElement) placeholderElement.style.display = 'flex';
        }
    }
    
    updateRecordingStatus() {
        const recordingElement = document.getElementById('recording-status');
        if (recordingElement) {
            recordingElement.className = this.isRecording ? 'status-indicator status-recording' : 'status-indicator status-offline';
        }
        
        const recordingTextElement = document.getElementById('recording-text');
        if (recordingTextElement) {
            recordingTextElement.textContent = this.isRecording ? '녹화 중' : '대기';
        }
        
        // 버튼 활성화/비활성화
        const startBtn = document.getElementById('start-recording');
        const stopBtn = document.getElementById('stop-recording');
        
        if (startBtn) startBtn.disabled = this.isRecording;
        if (stopBtn) stopBtn.disabled = !this.isRecording;
    }
    
    updateUptime() {
        this.stats.uptime += 1;
        const uptimeElement = document.getElementById('uptime-value');
        if (uptimeElement) {
            const hours = Math.floor(this.stats.uptime / 3600);
            const minutes = Math.floor((this.stats.uptime % 3600) / 60);
            const seconds = this.stats.uptime % 60;
            uptimeElement.textContent = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }
    
    showNotification(message, type = 'info') {
        // 간단한 알림 표시 (실제로는 toast 라이브러리 사용 권장)
        console.log(`${type.toUpperCase()}: ${message}`);
        
        // 임시 알림 요소 생성
        const notification = document.createElement('div');
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s ease;
            ${type === 'success' ? 'background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);' : ''}
            ${type === 'error' ? 'background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);' : ''}
            ${type === 'info' ? 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);' : ''}
        `;
        
        document.body.appendChild(notification);
        
        // 애니메이션 효과
        setTimeout(() => {
            notification.style.opacity = '1';
        }, 100);
        
        // 3초 후 제거
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
}

// 페이지 로드 시 대시보드 초기화
document.addEventListener('DOMContentLoaded', () => {
    window.cctvDashboard = new CCTVDashboard();
}); 