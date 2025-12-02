/**
 * Distilled Vision Agent - Improved Web Game Client
 * 
 * 개선 사항:
 * - 현대적인 그래픽 (그라데이션, 그림자, 애니메이션)
 * - 부드러운 키보드 조작 (연속 입력)
 * - 파티클 효과
 * - 데이터 수집 시스템 (훈련용)
 * 
 * Author: Minsuk Kim (mk4434) - Improved Version
 */

class ImprovedWebGameClient {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.socket = io();
        
        // 게임 상태
        this.gameState = null;
        this.currentMode = 'human';
        this.isGameRunning = false;
        this.lastFrameTime = 0;
        this.fps = 0;
        
        // 키보드 상태 (부드러운 조작을 위해)
        this.keys = {
            left: false,
            right: false,
            jump: false
        };
        
        // 애니메이션 & 그래픽
        this.particles = [];
        this.playerAnimation = {
            frame: 0,
            lastUpdate: 0,
            speed: 100 // ms
        };
        
        // 데이터 수집 (훈련용)
        this.gameplayData = {
            frames: [],
            actions: [],
            rewards: [],
            sessionId: null,
            startTime: null
        };
        
        this.initializeEventListeners();
        this.initializeSocketEvents();
        this.gameLoop();
        
        console.log('🎮 Improved Web Game Client initialized');
    }
    
    initializeEventListeners() {
        // ✅ 개선: 키보드 상태 추적 (연속 입력 지원)
        document.addEventListener('keydown', (e) => {
            if (!this.isGameRunning || this.currentMode !== 'human') return;
            
            switch(e.code) {
                case 'ArrowLeft':
                case 'KeyA':
                    this.keys.left = true;
                    e.preventDefault();
                    break;
                case 'ArrowRight':
                case 'KeyD':
                    this.keys.right = true;
                    e.preventDefault();
                    break;
                case 'ArrowUp':
                case 'Space':
                    if (!this.keys.jump) {  // 점프는 한 번만
                        this.keys.jump = true;
                        this.sendAction('jump');
                        this.createJumpParticles();
                        e.preventDefault();
                    }
                    break;
            }
        });
        
        document.addEventListener('keyup', (e) => {
            switch(e.code) {
                case 'ArrowLeft':
                case 'KeyA':
                    this.keys.left = false;
                    break;
                case 'ArrowRight':
                case 'KeyD':
                    this.keys.right = false;
                    break;
                case 'ArrowUp':
                case 'Space':
                    this.keys.jump = false;
                    break;
            }
        });
        
        // 버튼 이벤트
        document.getElementById('humanModeBtn')?.addEventListener('click', () => {
            this.startGame('human');
        });
        
        document.getElementById('aiModeBtn')?.addEventListener('click', () => {
            this.startGame('ai');
        });
        
        document.getElementById('restartBtn')?.addEventListener('click', () => {
            this.restartGame();
        });
    }
    
    initializeSocketEvents() {
        this.socket.on('connect', () => {
            console.log('🔗 서버에 연결됨');
        });
        
        this.socket.on('game_started', (data) => {
            console.log('🚀 게임 시작:', data);
            this.gameState = data;
            this.isGameRunning = true;
            
            // 데이터 수집 초기화
            this.gameplayData = {
                frames: [],
                actions: [],
                rewards: [],
                sessionId: data.session_id || Date.now().toString(),
                startTime: Date.now(),
                mode: this.currentMode
            };
        });
        
        this.socket.on('game_update', (data) => {
            this.gameState = data;
            
            // 데이터 수집
            this.collectGameplayData(data);
        });
        
        this.socket.on('game_over', (data) => {
            this.isGameRunning = false;
            this.showGameOver(data);
            
            // 게임 종료 시 데이터 저장
            this.saveGameplayData();
        });
    }
    
    sendAction(action) {
        if (!this.isGameRunning) return;
        
        this.socket.emit('player_action', { 
            action: action,
            timestamp: Date.now()
        });
        
        // 액션 기록
        this.gameplayData.actions.push({
            action: action,
            timestamp: Date.now(),
            gameState: this.gameState
        });
    }
    
    // ✅ 개선: 부드러운 연속 입력 처리
    processKeyboardInput() {
        if (!this.isGameRunning || this.currentMode !== 'human') return;
        
        // 좌우 이동은 연속으로 처리
        if (this.keys.left) {
            this.sendAction('left');
        } else if (this.keys.right) {
            this.sendAction('right');
        }
    }
    
    gameLoop(timestamp = 0) {
        requestAnimationFrame((ts) => this.gameLoop(ts));
        
        // FPS 계산
        const deltaTime = timestamp - this.lastFrameTime;
        this.fps = Math.round(1000 / deltaTime);
        this.lastFrameTime = timestamp;
        
        // 키보드 입력 처리 (매 프레임)
        this.processKeyboardInput();
        
        // 화면 그리기
        this.render();
        
        // 파티클 업데이트
        this.updateParticles(deltaTime);
    }
    
    // ✅ 개선: 현대적인 그래픽 렌더링
    render() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // 배경 (그라데이션)
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, '#1a1a2e');
        gradient.addColorStop(1, '#0f3460');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        
        // 별 효과
        this.drawStars();
        
        if (!this.gameState) return;
        
        // 플레이어 그리기 (현대적 디자인)
        if (this.gameState.player) {
            this.drawPlayer(this.gameState.player);
        }
        
        // 장애물 그리기 (현대적 디자인)
        if (this.gameState.obstacles) {
            this.gameState.obstacles.forEach(obs => {
                this.drawObstacle(obs);
            });
        }
        
        // 파티클 그리기
        this.particles.forEach(particle => {
            this.drawParticle(particle);
        });
        
        // UI 그리기
        this.drawUI();
    }
    
    drawPlayer(player) {
        const ctx = this.ctx;
        
        // 그림자
        ctx.save();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.beginPath();
        ctx.ellipse(player.x + 20, player.y + 45, 20, 5, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        
        // 플레이어 몸체 (그라데이션)
        const playerGradient = ctx.createRadialGradient(
            player.x + 20, player.y + 20, 5,
            player.x + 20, player.y + 20, 30
        );
        playerGradient.addColorStop(0, '#00d9ff');
        playerGradient.addColorStop(1, '#0099ff');
        
        ctx.fillStyle = playerGradient;
        ctx.shadowColor = '#00d9ff';
        ctx.shadowBlur = 15;
        
        // 애니메이션 효과
        const bounce = Math.sin(Date.now() / 200) * 2;
        
        ctx.beginPath();
        ctx.roundRect(player.x, player.y + bounce, 40, 40, 10);
        ctx.fill();
        
        // 눈
        ctx.fillStyle = '#ffffff';
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.arc(player.x + 12, player.y + 15 + bounce, 4, 0, Math.PI * 2);
        ctx.arc(player.x + 28, player.y + 15 + bounce, 4, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.restore();
    }
    
    drawObstacle(obstacle) {
        const ctx = this.ctx;
        
        // 그림자
        ctx.save();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.beginPath();
        ctx.ellipse(obstacle.x + 20, obstacle.y + 45, 20, 5, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        
        // 장애물 (위험한 느낌의 빨간색)
        const obsGradient = ctx.createRadialGradient(
            obstacle.x + 20, obstacle.y + 20, 5,
            obstacle.x + 20, obstacle.y + 20, 30
        );
        obsGradient.addColorStop(0, '#ff4757');
        obsGradient.addColorStop(1, '#ff0000');
        
        ctx.fillStyle = obsGradient;
        ctx.shadowColor = '#ff4757';
        ctx.shadowBlur = 20;
        
        // 회전 애니메이션
        const rotation = Date.now() / 1000;
        ctx.save();
        ctx.translate(obstacle.x + 20, obstacle.y + 20);
        ctx.rotate(rotation);
        ctx.fillRect(-20, -20, 40, 40);
        ctx.restore();
        
        ctx.restore();
    }
    
    drawStars() {
        const ctx = this.ctx;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        
        for (let i = 0; i < 50; i++) {
            const x = (i * 123) % this.canvas.width;
            const y = (i * 456) % this.canvas.height;
            const size = (i % 3) + 1;
            const twinkle = Math.sin(Date.now() / 500 + i) * 0.5 + 0.5;
            
            ctx.globalAlpha = twinkle;
            ctx.fillRect(x, y, size, size);
        }
        ctx.globalAlpha = 1;
    }
    
    drawUI() {
        const ctx = this.ctx;
        
        // 점수
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 24px Arial';
        ctx.shadowColor = '#000000';
        ctx.shadowBlur = 5;
        ctx.fillText(`Score: ${this.gameState?.score || 0}`, 20, 40);
        
        // 생존 시간
        const survivalTime = this.gameState?.survival_time || 0;
        ctx.fillText(`Time: ${survivalTime.toFixed(1)}s`, 20, 70);
        
        // FPS
        ctx.font = '16px Arial';
        ctx.fillStyle = this.fps >= 50 ? '#00ff00' : '#ff0000';
        ctx.fillText(`FPS: ${this.fps}`, this.canvas.width - 100, 30);
        
        // 모드 표시
        ctx.fillStyle = this.currentMode === 'human' ? '#00d9ff' : '#ff4757';
        ctx.fillText(`Mode: ${this.currentMode.toUpperCase()}`, this.canvas.width - 150, 60);
        
        ctx.shadowBlur = 0;
    }
    
    // 파티클 효과
    createJumpParticles() {
        if (!this.gameState?.player) return;
        
        for (let i = 0; i < 10; i++) {
            this.particles.push({
                x: this.gameState.player.x + 20,
                y: this.gameState.player.y + 40,
                vx: (Math.random() - 0.5) * 4,
                vy: Math.random() * 2,
                life: 1.0,
                color: '#00d9ff'
            });
        }
    }
    
    updateParticles(deltaTime) {
        this.particles = this.particles.filter(p => {
            p.x += p.vx;
            p.y += p.vy;
            p.life -= deltaTime / 1000;
            return p.life > 0;
        });
    }
    
    drawParticle(particle) {
        const ctx = this.ctx;
        ctx.save();
        ctx.globalAlpha = particle.life;
        ctx.fillStyle = particle.color;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }
    
    // 데이터 수집 시스템 (훈련용)
    collectGameplayData(gameState) {
        if (!this.gameplayData.startTime) return;
        
        this.gameplayData.frames.push({
            timestamp: Date.now(),
            gameState: {
                player: gameState.player,
                obstacles: gameState.obstacles,
                score: gameState.score,
                survival_time: gameState.survival_time
            }
        });
        
        // 메모리 관리: 최근 1000 프레임만 유지
        if (this.gameplayData.frames.length > 1000) {
            this.gameplayData.frames.shift();
        }
    }
    
    saveGameplayData() {
        if (this.gameplayData.frames.length === 0) return;
        
        const data = {
            ...this.gameplayData,
            endTime: Date.now(),
            duration: Date.now() - this.gameplayData.startTime,
            finalScore: this.gameState?.score || 0,
            finalSurvivalTime: this.gameState?.survival_time || 0
        };
        
        // 서버에 데이터 전송
        this.socket.emit('save_gameplay_data', data);
        
        console.log('📊 게임플레이 데이터 저장:', {
            frames: data.frames.length,
            actions: data.actions.length,
            duration: (data.duration / 1000).toFixed(1) + 's'
        });
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
    
    showGameOver(data) {
        const overlay = document.getElementById('gameOverlay');
        if (overlay) {
            overlay.style.display = 'flex';
            overlay.innerHTML = `
                <div style="background: rgba(0,0,0,0.9); padding: 40px; border-radius: 20px; text-align: center;">
                    <h1 style="color: #ff4757; font-size: 48px; margin-bottom: 20px;">Game Over!</h1>
                    <p style="color: #ffffff; font-size: 24px;">Score: ${data.score || 0}</p>
                    <p style="color: #ffffff; font-size: 20px;">Time: ${(data.survival_time || 0).toFixed(1)}s</p>
                    <button id="restartBtn" style="margin-top: 20px; padding: 15px 30px; font-size: 20px; background: #00d9ff; border: none; border-radius: 10px; cursor: pointer;">
                        Restart
                    </button>
                </div>
            `;
            
            document.getElementById('restartBtn').addEventListener('click', () => {
                overlay.style.display = 'none';
                this.restartGame();
            });
        }
    }
}

// 페이지 로드 시 게임 클라이언트 초기화
document.addEventListener('DOMContentLoaded', () => {
    window.gameClient = new ImprovedWebGameClient();
    console.log('🎮 Improved Game Client Ready!');
});
