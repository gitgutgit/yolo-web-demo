/**
 * Distilled Vision Agent - Web Game Client
 * 
 * HTML5 Canvas 기반 게임 클라이언트
 * SocketIO를 통한 실시간 통신
 * 
 * Author: Minsuk Kim (mk4434)
 */

class WebGameClient {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.socket = io();
        
        this.gameState = null;
        this.currentMode = 'human';
        this.isGameRunning = false;
        this.lastFrameTime = 0;
        this.fps = 0;
        
        this.keys = {};
        
        this.initializeEventListeners();
        this.initializeSocketEvents();
        this.loadLeaderboard();
        
        console.log('🎮 Web Game Client initialized');
    }
    
    initializeEventListeners() {
        // 키보드 이벤트
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
        document.addEventListener('keyup', (e) => this.handleKeyUp(e));
        
        // 버튼 이벤트
        document.getElementById('humanModeBtn').addEventListener('click', () => {
            this.startGame('human');
        });
        
        document.getElementById('aiModeBtn').addEventListener('click', () => {
            this.startGame('ai');
        });
        
        document.getElementById('restartBtn').addEventListener('click', () => {
            this.restartGame();
        });
        
        document.getElementById('switchModeBtn').addEventListener('click', () => {
            this.showStartScreen();
        });
        
        // 게임 루프 시작
        this.gameLoop();
    }
    
    initializeSocketEvents() {
        this.socket.on('connect', () => {
            console.log('🔗 서버에 연결됨');
            this.updateConnectionStatus('Connected', 'connected');
        });
        
        this.socket.on('disconnect', () => {
            console.log('❌ 서버 연결 끊김');
            this.updateConnectionStatus('Disconnected', 'disconnected');
        });
        
        this.socket.on('connected', (data) => {
            console.log('✅ 게임 세션 생성:', data.session_id);
        });
        
        this.socket.on('game_started', (data) => {
            console.log('🚀 게임 시작:', data);
            this.gameState = data.state;
            this.isGameRunning = true;
            this.hideOverlay();
            this.updateUI();
        });
        
        this.socket.on('game_update', (data) => {
            this.gameState = data.state;
            this.updateUI();
            
            // AI 액션 표시
            if (data.ai_action && this.currentMode === 'ai') {
                this.showAIAction(data.ai_action);
            }
        });
        
        this.socket.on('mode_switched', (data) => {
            console.log('🔄 모드 전환:', data.mode);
            this.currentMode = data.mode;
            this.gameState = data.state;
            this.updateModeUI();
        });
        
        this.socket.on('error', (error) => {
            console.error('❌ 소켓 오류:', error);
        });
    }
    
    handleKeyDown(e) {
        this.keys[e.code] = true;
        
        // 게임 단축키
        if (e.code === 'KeyH') {
            this.switchMode('human');
        } else if (e.code === 'KeyI') {
            this.switchMode('ai');
        } else if (e.code === 'KeyR') {
            this.restartGame();
        }
        
        // Human 모드 게임 컨트롤
        if (this.isGameRunning && this.currentMode === 'human') {
            let action = null;
            
            if (e.code === 'Space' || e.code === 'ArrowUp') {
                action = 'jump';
                e.preventDefault();
            } else if (e.code === 'KeyA' || e.code === 'ArrowLeft') {
                action = 'left';
            } else if (e.code === 'KeyD' || e.code === 'ArrowRight') {
                action = 'right';
            }
            
            if (action) {
                this.socket.emit('player_action', { action: action });
            }
        }
    }
    
    handleKeyUp(e) {
        this.keys[e.code] = false;
    }
    
    startGame(mode) {
        console.log(`🎮 게임 시작: ${mode} 모드`);
        this.currentMode = mode;
        this.socket.emit('start_game', { mode: mode });
    }
    
    restartGame() {
        console.log('🔄 게임 재시작');
        this.socket.emit('start_game', { mode: this.currentMode });
    }
    
    switchMode(mode) {
        if (mode !== this.currentMode) {
            console.log(`🔄 모드 전환: ${this.currentMode} -> ${mode}`);
            this.currentMode = mode;
            this.socket.emit('switch_mode', { mode: mode });
        }
    }
    
    showStartScreen() {
        document.getElementById('gameOverlay').style.display = 'flex';
        document.getElementById('startScreen').style.display = 'block';
        document.getElementById('gameOverScreen').style.display = 'none';
        this.isGameRunning = false;
    }
    
    hideOverlay() {
        document.getElementById('gameOverlay').style.display = 'none';
    }
    
    showGameOver() {
        if (!this.gameState) return;
        
        const finalStats = document.getElementById('finalStats');
        finalStats.innerHTML = `
            <div class="final-stat">
                <strong>Mode:</strong> ${this.currentMode.toUpperCase()}
            </div>
            <div class="final-stat">
                <strong>Final Score:</strong> ${this.gameState.score}
            </div>
            <div class="final-stat">
                <strong>Survival Time:</strong> ${this.gameState.survival_time.toFixed(1)}s
            </div>
            <div class="final-stat">
                <strong>Frames:</strong> ${this.gameState.frame_count}
            </div>
        `;
        
        document.getElementById('gameOverlay').style.display = 'flex';
        document.getElementById('startScreen').style.display = 'none';
        document.getElementById('gameOverScreen').style.display = 'block';
        
        this.isGameRunning = false;
    }
    
    updateUI() {
        if (!this.gameState) return;
        
        // 게임 정보 업데이트
        document.getElementById('currentMode').textContent = this.currentMode.toUpperCase();
        document.getElementById('currentScore').textContent = this.gameState.score;
        document.getElementById('currentTime').textContent = `${this.gameState.survival_time.toFixed(1)}s`;
        document.getElementById('frameCounter').textContent = this.gameState.frame_count;
        document.getElementById('obstacleCount').textContent = this.gameState.obstacles.length;
        
        // 게임 오버 체크
        if (this.gameState.game_over && this.isGameRunning) {
            this.showGameOver();
        }
    }
    
    updateModeUI() {
        const humanControls = document.getElementById('humanControls');
        const aiControls = document.getElementById('aiControls');
        const aiActionInfo = document.getElementById('aiActionInfo');
        
        if (this.currentMode === 'human') {
            humanControls.style.display = 'block';
            aiControls.style.display = 'none';
            aiActionInfo.style.display = 'none';
        } else {
            humanControls.style.display = 'none';
            aiControls.style.display = 'block';
            aiActionInfo.style.display = 'flex';
        }
    }
    
    showAIAction(action) {
        const aiActionElement = document.getElementById('aiAction');
        aiActionElement.textContent = action.toUpperCase();
        aiActionElement.classList.add('active');
        
        setTimeout(() => {
            aiActionElement.classList.remove('active');
        }, 300);
    }
    
    updateConnectionStatus(status, className) {
        const statusElement = document.getElementById('connectionStatus');
        statusElement.textContent = status;
        statusElement.className = `stat-value status-${className}`;
    }
    
    render() {
        if (!this.gameState) return;
        
        // 캔버스 클리어
        this.ctx.fillStyle = '#f0f8ff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 플레이어 그리기
        const playerColor = this.currentMode === 'human' ? '#4facfe' : '#43e97b';
        this.ctx.fillStyle = playerColor;
        this.ctx.fillRect(
            this.gameState.player.x,
            this.gameState.player.y,
            GAME_CONFIG.player_size,
            GAME_CONFIG.player_size
        );
        
        // 플레이어 테두리
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(
            this.gameState.player.x,
            this.gameState.player.y,
            GAME_CONFIG.player_size,
            GAME_CONFIG.player_size
        );
        
        // 장애물 그리기
        this.ctx.fillStyle = '#ff6b6b';
        this.gameState.obstacles.forEach(obstacle => {
            this.ctx.fillRect(
                obstacle.x,
                obstacle.y,
                GAME_CONFIG.obstacle_size,
                GAME_CONFIG.obstacle_size
            );
            
            // 장애물 테두리
            this.ctx.strokeStyle = '#d63031';
            this.ctx.strokeRect(
                obstacle.x,
                obstacle.y,
                GAME_CONFIG.obstacle_size,
                GAME_CONFIG.obstacle_size
            );
        });
        
        // 모드 표시
        this.ctx.fillStyle = '#333';
        this.ctx.font = '16px Arial';
        this.ctx.fillText(`Mode: ${this.currentMode.toUpperCase()}`, 10, 25);
        
        // 점수 표시
        this.ctx.fillText(`Score: ${this.gameState.score}`, 10, 45);
        this.ctx.fillText(`Time: ${this.gameState.survival_time.toFixed(1)}s`, 10, 65);
    }
    
    calculateFPS() {
        const now = performance.now();
        const delta = now - this.lastFrameTime;
        this.fps = Math.round(1000 / delta);
        this.lastFrameTime = now;
        
        document.getElementById('fpsCounter').textContent = this.fps;
    }
    
    gameLoop() {
        this.calculateFPS();
        this.render();
        requestAnimationFrame(() => this.gameLoop());
    }
    
    async loadLeaderboard() {
        try {
            const response = await fetch('/api/leaderboard');
            const leaderboard = await response.json();
            
            const leaderboardElement = document.getElementById('leaderboard');
            leaderboardElement.innerHTML = '';
            
            leaderboard.forEach((entry, index) => {
                const item = document.createElement('div');
                item.className = `leaderboard-item ${entry.mode}`;
                
                item.innerHTML = `
                    <div class="player-info">
                        <div class="player-name">#${index + 1} ${entry.name}</div>
                        <div class="player-mode">${entry.mode.toUpperCase()}</div>
                    </div>
                    <div class="player-stats">
                        <div class="player-score">${entry.score}</div>
                        <div class="player-time">${entry.time}s</div>
                    </div>
                `;
                
                leaderboardElement.appendChild(item);
            });
            
        } catch (error) {
            console.error('리더보드 로드 실패:', error);
            document.getElementById('leaderboard').innerHTML = 
                '<div class="loading">Failed to load leaderboard</div>';
        }
    }
}

// 게임 클라이언트 초기화
document.addEventListener('DOMContentLoaded', () => {
    console.log('🌐 DOM 로드 완료, 게임 클라이언트 시작');
    window.gameClient = new WebGameClient();
});

// 페이지 언로드 시 정리
window.addEventListener('beforeunload', () => {
    if (window.gameClient && window.gameClient.socket) {
        window.gameClient.socket.disconnect();
    }
});
