#!/usr/bin/env python3
"""
YOLO-You-Only-Live-Once 게임 서버

YOLO + PPO 통합:
- 서버에서 프레임 렌더링 → YOLO 추론 → Detection 전송
- 클라이언트에서 Detection Box 오버레이

수정: 2025-11-29
- YOLO 실시간 탐지 통합
- Detection 정보 클라이언트 전송
- PPO 모델 기반 AI
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import time
import random
import threading
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 없음")

# YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics 없음 - YOLO 사용 불가")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'game-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ============================================================================
# 게임 설정
# ============================================================================

WIDTH = 960
HEIGHT = 720
PLAYER_SIZE = 50
OBSTACLE_SIZE = 50

# YOLO 클래스 매핑
YOLO_CLASSES = {
    0: 'player',
    1: 'meteor',
    2: 'star',
    3: 'caution_lava',
    4: 'exist_lava'
}

# 액션
ACTION_LIST = ["stay", "left", "right", "jump"]
STATE_DIM = 26

# 객체 타입 정의
OBJECT_TYPES = {
    'meteor': {'color': '#FF4444', 'size': 50, 'vy': 5, 'score': 0, 'reward': -100},
    'star': {'color': '#FFD700', 'size': 30, 'vy': 3, 'score': 10, 'reward': 20}
}

# 용암 설정
LAVA_CONFIG = {
    'enabled': True,
    'warning_duration': 3.0,
    'active_duration': 3.0,
    'interval': 20.0,
    'height': 120,
    'damage_per_frame': 3,
    'zone_width': 320
}

# ============================================================================
# 모델 로드
# ============================================================================

project_root = Path(__file__).parent.parent
yolo_model_path = os.getenv('YOLO_MODEL_PATH', str(project_root / 'AI_model' / 'yolo_fine.pt'))
ppo_model_path = os.getenv('PPO_MODEL_PATH', str(project_root / 'web_app' / 'models' / 'rl' / 'ppo_agent.pt'))

# YOLO 모델
yolo_model = None
if YOLO_AVAILABLE and Path(yolo_model_path).exists():
    try:
        yolo_model = YOLO(yolo_model_path)
        print(f"✅ YOLO 모델 로드: {yolo_model_path}")
    except Exception as e:
        print(f"⚠️ YOLO 로드 실패: {e}")

# PPO 모델
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=26, action_dim=4, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=-1)

ppo_model = None
ppo_device = None
if TORCH_AVAILABLE and Path(ppo_model_path).exists():
    try:
        ppo_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(ppo_model_path, map_location=ppo_device)
        ppo_model = PolicyNetwork(STATE_DIM, len(ACTION_LIST)).to(ppo_device)
        
        if 'policy_state_dict' in checkpoint:
            ppo_model.load_state_dict(checkpoint['policy_state_dict'])
        elif 'policy' in checkpoint:
            ppo_model.load_state_dict(checkpoint['policy'])
        else:
            ppo_model.load_state_dict(checkpoint)
        
        ppo_model.eval()
        print(f"✅ PPO 모델 로드: {ppo_model_path}")
    except Exception as e:
        print(f"⚠️ PPO 로드 실패: {e}")

print(f"🤖 AI 시스템: YOLO={'✅' if yolo_model else '❌'}, PPO={'✅' if ppo_model else '❌'}")

# ============================================================================
# State Encoder
# ============================================================================

def encode_state(detections: list, game_state: dict = None) -> np.ndarray:
    """YOLO detections → 26-dim state vector"""
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    
    # Find Player
    player_x, player_y = 0.5, 0.5
    for d in detections:
        if d['cls'] == 0:
            player_x = d['x']
            player_y = d['y']
            vec[0] = player_x
            vec[1] = player_y
            if player_y >= 0.90:
                vec[23] = 1.0
            break
    
    # Fallback
    if vec[0] == 0 and game_state:
        player_x = (game_state.get('player_x', WIDTH//2) + PLAYER_SIZE/2) / WIDTH
        player_y = (game_state.get('player_y', HEIGHT//2) + PLAYER_SIZE/2) / HEIGHT
        vec[0] = player_x
        vec[1] = player_y
    
    # Meteors
    meteors = []
    for d in detections:
        if d['cls'] == 1:
            dx = d['x'] - player_x
            dy = d['y'] - player_y
            dist = np.sqrt(dx**2 + dy**2)
            meteors.append((dist, dx, dy, 0.0, 0.5))
    
    meteors.sort(key=lambda x: x[0])
    for i in range(3):
        base_idx = 2 + i * 5
        if i < len(meteors):
            dist, dx, dy, vx, vy = meteors[i]
            vec[base_idx] = np.clip(dx, -1, 1)
            vec[base_idx+1] = np.clip(dy, -1, 1)
            vec[base_idx+2] = np.clip(dist, 0, 1.5)
            vec[base_idx+3] = vx
            vec[base_idx+4] = vy
        else:
            vec[base_idx+2] = 1.0
    
    # Star
    nearest_star = 1.0
    for d in detections:
        if d['cls'] == 2:
            dx = d['x'] - player_x
            dy = d['y'] - player_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < nearest_star:
                nearest_star = dist
                vec[17] = np.clip(dx, -1, 1)
                vec[18] = np.clip(dy, -1, 1)
                vec[19] = np.clip(dist, 0, 1)
    if nearest_star == 1.0:
        vec[19] = 1.0
    
    # Lava
    for d in detections:
        if d['cls'] == 3:
            vec[20] = 1.0
            vec[22] = np.clip(d['x'] - player_x, -1, 1)
        elif d['cls'] == 4:
            vec[21] = 1.0
            vec[22] = np.clip(d['x'] - player_x, -1, 1)
    
    return vec


# ============================================================================
# Game Class
# ============================================================================

games = {}

class Game:
    def __init__(self, sid):
        self.sid = sid
        self.reset()
        self.show_detections = True  # Detection Box 표시 여부 (기본값: True)
        
    def reset(self):
        self.player_x = WIDTH // 2
        self.player_y = HEIGHT // 2
        self.player_vy = 0
        self.obstacles = []
        self.score = 0
        self.running = False
        self.mode = "human"
        self.player_name = None
        self.start_time = time.time()
        self.frame = 0
        self.game_over = False
        self.player_health = 100
        
        # Lava
        self.lava_state = 'inactive'
        self.lava_timer = LAVA_CONFIG['interval']
        self.lava_phase_timer = 0
        self.lava_zone_x = 0
        
        # AI 관련
        self.ai_level = 2
        self.last_detections = []
        self.last_action_probs = [0.25, 0.25, 0.25, 0.25]
        
    def render_frame(self) -> np.ndarray:
        """서버에서 프레임 렌더링 (YOLO 입력용)"""
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # Background - dark blue gradient
        for y in range(HEIGHT):
            ratio = y / HEIGHT
            b = int(40 + ratio * 20)
            g = int(20 + ratio * 10)
            r = int(60 + ratio * 30)
            img[y, :] = (b, g, r)
        
        # Stars background
        np.random.seed(42)
        for _ in range(100):
            sx = np.random.randint(0, WIDTH)
            sy = np.random.randint(0, HEIGHT)
            brightness = np.random.randint(100, 255)
            cv2.circle(img, (sx, sy), 1, (brightness, brightness, brightness), -1)
        
        # Lava
        if self.lava_state == 'warning':
            x1 = self.lava_zone_x
            x2 = x1 + LAVA_CONFIG['zone_width']
            y1 = HEIGHT - LAVA_CONFIG['height']
            
            # Warning overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, HEIGHT), (0, 165, 255), -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            
            # Warning border
            cv2.rectangle(img, (x1, y1), (x2, HEIGHT), (0, 165, 255), 3)
            
            # Warning text
            cv2.putText(img, "WARNING", (x1 + 100, y1 + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            
        elif self.lava_state == 'active':
            x1 = self.lava_zone_x
            x2 = x1 + LAVA_CONFIG['zone_width']
            y1 = HEIGHT - LAVA_CONFIG['height']
            
            # Lava gradient
            for row in range(y1, HEIGHT):
                ratio = (row - y1) / LAVA_CONFIG['height']
                r = int(255 - ratio * 35)
                g = int(69 + ratio * 30)
                b = int(0 + ratio * 60)
                cv2.line(img, (x1, row), (x2, row), (b, g, r), 1)
            
            # Bubbles
            for i in range(8):
                bx = x1 + 40 + i * 35 + int(np.sin(self.frame * 0.1 + i) * 10)
                by = y1 + 30 + int(np.sin(self.frame * 0.15 + i * 0.5) * 20)
                cv2.circle(img, (bx, by), 8, (0, 140, 255), -1)
            
            # FIRE LAVA text
            cv2.putText(img, "FIRE LAVA", (x1 + 80, y1 + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Border
            cv2.rectangle(img, (x1, y1), (x2, HEIGHT), (0, 0, 255), 3)
        
        # Obstacles
        for obs in self.obstacles:
            ox, oy = int(obs['x']), int(obs['y'])
            size = obs.get('size', OBSTACLE_SIZE)
            obj_type = obs.get('type', 'meteor')
            
            if obj_type == 'meteor':
                # Meteor - red with tail
                cv2.circle(img, (ox + size//2, oy + size//2), size//2, (0, 0, 200), -1)
                cv2.circle(img, (ox + size//2, oy + size//2), size//2 - 5, (0, 0, 255), -1)
                
                # Tail
                for i in range(3):
                    ty = oy - 10 - i * 8
                    alpha = 0.7 - i * 0.2
                    tail_color = (0, int(100 * alpha), int(255 * alpha))
                    cv2.circle(img, (ox + size//2, ty), size//4 - i*2, tail_color, -1)
            else:
                # Star - yellow
                cx, cy = ox + size//2, oy + size//2
                pts = []
                for i in range(5):
                    angle = i * 72 - 90
                    rad = np.radians(angle)
                    pts.append((int(cx + size//2 * np.cos(rad)), int(cy + size//2 * np.sin(rad))))
                    rad2 = np.radians(angle + 36)
                    pts.append((int(cx + size//4 * np.cos(rad2)), int(cy + size//4 * np.sin(rad2))))
                pts = np.array(pts, np.int32)
                cv2.fillPoly(img, [pts], (0, 255, 255))
                
                # Glow
                cv2.circle(img, (cx, cy), size//2 + 5, (0, 200, 255), 2)
        
        # Player
        px, py = int(self.player_x), int(self.player_y)
        
        # Body
        cv2.rectangle(img, (px, py), (px + PLAYER_SIZE, py + PLAYER_SIZE), (255, 200, 0), -1)
        cv2.rectangle(img, (px + 5, py + 5), (px + PLAYER_SIZE - 5, py + PLAYER_SIZE - 5), (255, 220, 50), -1)
        
        # Eyes
        cv2.circle(img, (px + 15, py + 18), 6, (255, 255, 255), -1)
        cv2.circle(img, (px + 35, py + 18), 6, (255, 255, 255), -1)
        cv2.circle(img, (px + 15, py + 18), 3, (0, 0, 0), -1)
        cv2.circle(img, (px + 35, py + 18), 3, (0, 0, 0), -1)
        
        # Smile
        cv2.ellipse(img, (px + 25, py + 35), (10, 5), 0, 0, 180, (0, 0, 0), 2)
        
        return img
    
    def run_yolo(self, frame: np.ndarray) -> list:
        """YOLO 추론"""
        if yolo_model is None:
            return self._fallback_detections()
        
        try:
            results = yolo_model(frame, verbose=False)
            
            detections = []
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    x, y, w, h = box.xywhn[0].tolist()
                    conf = float(box.conf[0])
                    
                    # 픽셀 좌표 계산
                    xyxy = box.xyxy[0].tolist()
                    
                    detections.append({
                        'cls': cls,
                        'class_name': YOLO_CLASSES.get(cls, f'cls{cls}'),
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'conf': conf,
                        'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    })
            
            return detections
            
        except Exception as e:
            print(f"⚠️ YOLO 오류: {e}")
            return self._fallback_detections()
    
    def _fallback_detections(self) -> list:
        """YOLO 없을 때 game_state에서 detection 생성"""
        detections = []
        
        # Player
        detections.append({
            'cls': 0,
            'class_name': 'player',
            'x': (self.player_x + PLAYER_SIZE/2) / WIDTH,
            'y': (self.player_y + PLAYER_SIZE/2) / HEIGHT,
            'w': PLAYER_SIZE / WIDTH,
            'h': PLAYER_SIZE / HEIGHT,
            'conf': 1.0,
            'bbox': [int(self.player_x), int(self.player_y), 
                    int(self.player_x + PLAYER_SIZE), int(self.player_y + PLAYER_SIZE)]
        })
        
        # Obstacles
        for obs in self.obstacles:
            cls = 1 if obs.get('type') == 'meteor' else 2
            size = obs.get('size', 50)
            detections.append({
                'cls': cls,
                'class_name': 'meteor' if cls == 1 else 'star',
                'x': (obs['x'] + size/2) / WIDTH,
                'y': (obs['y'] + size/2) / HEIGHT,
                'w': size / WIDTH,
                'h': size / HEIGHT,
                'conf': 1.0,
                'bbox': [int(obs['x']), int(obs['y']), 
                        int(obs['x'] + size), int(obs['y'] + size)]
            })
        
        # Lava
        if self.lava_state in ['warning', 'active']:
            cls = 3 if self.lava_state == 'warning' else 4
            y1 = HEIGHT - LAVA_CONFIG['height']
            detections.append({
                'cls': cls,
                'class_name': 'caution_lava' if cls == 3 else 'exist_lava',
                'x': (self.lava_zone_x + LAVA_CONFIG['zone_width']/2) / WIDTH,
                'y': (y1 + LAVA_CONFIG['height']/2) / HEIGHT,
                'w': LAVA_CONFIG['zone_width'] / WIDTH,
                'h': LAVA_CONFIG['height'] / HEIGHT,
                'conf': 1.0,
                'bbox': [self.lava_zone_x, y1, 
                        self.lava_zone_x + LAVA_CONFIG['zone_width'], HEIGHT]
            })
        
        return detections
    
    def update(self):
        """게임 업데이트"""
        if self.game_over:
            return
        
        # 중력
        self.player_vy += 1
        self.player_y += self.player_vy
        
        # 바닥 충돌
        if self.player_y >= HEIGHT - PLAYER_SIZE:
            self.player_y = HEIGHT - PLAYER_SIZE
            self.player_vy = 0
        
        # 장애물 이동
        for obs in self.obstacles:
            obs['x'] += obs.get('vx', 0)
            obs['y'] += obs.get('vy', 5)
            
            if obs['x'] < -obs.get('size', OBSTACLE_SIZE):
                obs['x'] = WIDTH
            elif obs['x'] > WIDTH:
                obs['x'] = -obs.get('size', OBSTACLE_SIZE)
        
        # 화면 밖 장애물 제거
        before_count = len(self.obstacles)
        self.obstacles = [o for o in self.obstacles if o['y'] < HEIGHT]
        self.score += before_count - len(self.obstacles)
        
        # 충돌 검사
        self.check_collisions()
        
        # 새 객체 생성
        if random.random() < 0.05:
            obj_type = 'star' if random.random() < 0.1 else 'meteor'
            obj_config = OBJECT_TYPES[obj_type]
            self.obstacles.append({
                'type': obj_type,
                'x': random.randint(0, WIDTH - obj_config['size']),
                'y': -obj_config['size'],
                'vx': random.randint(-2, 2),
                'vy': obj_config['vy'],
                'size': obj_config['size']
            })
        
        # 용암 업데이트
        if LAVA_CONFIG['enabled']:
            self.update_lava()
        
        self.frame += 1
    
    def update_lava(self):
        """용암 업데이트"""
        dt = 1.0 / 30.0
        
        if self.lava_state == 'inactive':
            self.lava_timer -= dt
            if self.lava_timer <= 0:
                self.lava_state = 'warning'
                self.lava_phase_timer = LAVA_CONFIG['warning_duration']
                self.lava_zone_x = random.choice([0, WIDTH // 3, (WIDTH // 3) * 2])
        
        elif self.lava_state == 'warning':
            self.lava_phase_timer -= dt
            if self.lava_phase_timer <= 0:
                self.lava_state = 'active'
                self.lava_phase_timer = LAVA_CONFIG['active_duration']
        
        elif self.lava_state == 'active':
            self.lava_phase_timer -= dt
            
            # 데미지
            lava_x_end = self.lava_zone_x + LAVA_CONFIG['zone_width']
            lava_y = HEIGHT - LAVA_CONFIG['height']
            
            if (self.player_x + PLAYER_SIZE > self.lava_zone_x and 
                self.player_x < lava_x_end and 
                self.player_y + PLAYER_SIZE > lava_y):
                self.player_health -= LAVA_CONFIG['damage_per_frame']
                if self.player_health <= 0:
                    self.game_over = True
            
            if self.lava_phase_timer <= 0:
                self.lava_state = 'inactive'
                self.lava_timer = LAVA_CONFIG['interval']
                self.player_health = 100
    
    def check_collisions(self):
        """충돌 검사"""
        for obs in self.obstacles[:]:
            obj_size = obs.get('size', OBSTACLE_SIZE)
            
            if (self.player_x < obs['x'] + obj_size and
                self.player_x + PLAYER_SIZE > obs['x'] and
                self.player_y < obs['y'] + obj_size and
                self.player_y + PLAYER_SIZE > obs['y']):
                
                obj_type = obs.get('type', 'meteor')
                
                if obj_type == 'meteor':
                    self.game_over = True
                    self.running = False
                elif obj_type == 'star':
                    self.score += OBJECT_TYPES['star']['score']
                    self.obstacles.remove(obs)
    
    def jump(self):
        if self.player_y >= HEIGHT - PLAYER_SIZE - 5:
            self.player_vy = -18
    
    def move_left(self):
        self.player_x = max(0, self.player_x - 10)
    
    def move_right(self):
        self.player_x = min(WIDTH - PLAYER_SIZE, self.player_x + 10)
    
    def get_state(self):
        """클라이언트로 전송할 상태"""
        return {
            'player': {
                'x': self.player_x,
                'y': self.player_y,
                'vy': self.player_vy,
                'size': PLAYER_SIZE,
                'health': self.player_health
            },
            'obstacles': self.obstacles,
            'score': self.score,
            'time': time.time() - self.start_time,
            'frame': self.frame,
            'mode': self.mode,
            'game_over': self.game_over,
            'lava': {
                'state': self.lava_state,
                'timer': self.lava_phase_timer if self.lava_state != 'inactive' else self.lava_timer,
                'height': LAVA_CONFIG['height'],
                'zone_x': self.lava_zone_x,
                'zone_width': LAVA_CONFIG['zone_width']
            },
            # AI 정보
            'detections': self.last_detections if self.show_detections else [],
            'action_probs': self.last_action_probs,
            'ai_active': self.mode == 'ai'
        }


# ============================================================================
# AI Decision
# ============================================================================

def ai_decision(game: Game) -> str:
    """AI 의사결정 (YOLO + PPO)"""
    
    # 1) 프레임 렌더링
    frame = game.render_frame()
    
    # 2) YOLO 추론
    detections = game.run_yolo(frame)
    game.last_detections = detections
    
    # 3) State Encoding
    state_vec = encode_state(detections, {
        'player_x': game.player_x,
        'player_y': game.player_y
    })
    
    # 4) PPO Policy
    if ppo_model is not None:
        try:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(ppo_device)
                action_probs = ppo_model(state_tensor)
                probs = action_probs.cpu().numpy()[0]
                action_idx = np.argmax(probs)
            
            game.last_action_probs = probs.tolist()
            return ACTION_LIST[action_idx]
            
        except Exception as e:
            print(f"⚠️ PPO 오류: {e}")
    
    # Fallback: 휴리스틱
    return fallback_heuristic(game)


def fallback_heuristic(game: Game) -> str:
    """휴리스틱 폴백"""
    player_cx = game.player_x + PLAYER_SIZE / 2
    
    for obs in game.obstacles:
        if obs.get('type') != 'meteor':
            continue
        obs_cx = obs['x'] + obs.get('size', 50) / 2
        if obs['y'] < game.player_y and abs(obs_cx - player_cx) < 100:
            return 'right' if obs_cx < player_cx else 'left'
    
    return 'stay'


# ============================================================================
# Game Loop
# ============================================================================

def game_loop(sid):
    """게임 루프"""
    game = games.get(sid)
    if not game:
        return
    
    print(f"🎮 게임 시작: {sid} (모드: {game.mode})")
    
    while game.running and not game.game_over:
        try:
            # AI 모드
            if game.mode == 'ai':
                action = ai_decision(game)
                if action == 'jump':
                    game.jump()
                elif action == 'left':
                    game.move_left()
                elif action == 'right':
                    game.move_right()
            
            # Human 모드지만 Detection 표시가 켜져있는 경우 YOLO 실행
            elif game.show_detections:
                try:
                    frame = game.render_frame()
                    detections = game.run_yolo(frame)
                    game.last_detections = detections
                except Exception as e:
                    print(f"⚠️ YOLO Visualization Error: {e}")
            
            game.update()
            
            # 상태 전송 (detections 포함!)
            socketio.emit('game_update', {
                'state': game.get_state()
            }, room=sid)
            
            time.sleep(1.0 / 30)
            
        except Exception as e:
            print(f"❌ 에러: {e}")
            break
    
    # 게임 오버
    if game.game_over:
        survival_time = time.time() - game.start_time
        socketio.emit('game_over', {
            'score': game.score,
            'time': survival_time,
            'frame': game.frame,
            'mode': game.mode
        }, room=sid)
        print(f"🏁 게임 오버: {game.score}점, {survival_time:.1f}초")


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def api_stats():
    return jsonify({
        'yolo_available': yolo_model is not None,
        'ppo_available': ppo_model is not None,
        'active_games': len(games)
    })


# ============================================================================
# Socket Events
# ============================================================================

@socketio.on('connect')
def on_connect():
    from flask import request
    sid = request.sid
    games[sid] = Game(sid)
    print(f"✅ 연결: {sid}")
    emit('connected', {'config': {'width': WIDTH, 'height': HEIGHT}})

@socketio.on('disconnect')
def on_disconnect():
    from flask import request
    sid = request.sid
    if sid in games:
        games[sid].running = False
        del games[sid]
    print(f"❌ 연결 해제: {sid}")

@socketio.on('start_game')
def on_start_game(data):
    from flask import request
    sid = request.sid
    game = games.get(sid)
    
    if not game:
        return
    
    game.reset()
    game.mode = data.get('mode', 'human')
    game.player_name = data.get('player_name', None)
    game.ai_level = data.get('ai_level', 2)
    game.running = True
    
    print(f"🚀 게임 시작: {sid}, 모드: {game.mode}")
    
    thread = threading.Thread(target=game_loop, args=(sid,))
    thread.daemon = True
    thread.start()
    
    emit('game_started', {'state': game.get_state()})

@socketio.on('player_action')
def on_action(data):
    from flask import request
    sid = request.sid
    game = games.get(sid)
    
    if not game or not game.running:
        return
    
    action = data.get('action')
    if action == 'jump':
        game.jump()
    elif action == 'left':
        game.move_left()
    elif action == 'right':
        game.move_right()

@socketio.on('toggle_detections')
def on_toggle_detections():
    """G키: Detection 표시 토글"""
    from flask import request
    sid = request.sid
    game = games.get(sid)
    
    if game:
        game.show_detections = not game.show_detections
        print(f"🔲 Detection Box: {'ON' if game.show_detections else 'OFF'}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('DEBUG', 'True') == 'True'
    
    print("🎮 YOLO-You-Only-Live-Once 서버 시작!")
    print(f"🌐 http://localhost:{port}")
    print(f"🤖 YOLO: {'✅' if yolo_model else '❌'}")
    print(f"🧠 PPO: {'✅' if ppo_model else '❌'}")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)
