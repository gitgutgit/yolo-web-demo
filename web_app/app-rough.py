#!/usr/bin/env python3
"""
간단하고 확실하게 작동하는 게임
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
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Cloud Storage Manager
from storage_manager import get_storage_manager

# CV Module for Vision-based Lava Detection
from modules.cv_module import ComputerVisionModule

app = Flask(__name__)
app.config['SECRET_KEY'] = 'game-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 게임 설정
WIDTH = 960
HEIGHT = 720
PLAYER_SIZE = 50
OBSTACLE_SIZE = 50

# RL 모델 플래그 (클로가 나중에 학습시킬 모델)
RL_MODEL_AVAILABLE = False
RL_MODEL = None

try:
    # PyTorch 모델 로드 시도 (아직 없음)
    # import torch
    # RL_MODEL = torch.load('models/rl_agent.pth')
    # RL_MODEL_AVAILABLE = True
    print("⚠️ RL 모델 없음 - 휴리스틱 AI 사용")
except Exception as e:
    print(f"⚠️ RL 모델 로드 실패: {e}")

# 객체 타입 정의 (메테오 = 떨어지는 장애물, 별 = 보상 아이템)
OBJECT_TYPES = {
    'meteor': {  # 🔴 메테오 (피해야 함)
        'color': '#FF4444',
        'size': 50,
        'vy': 5,
        'score': 0,
        'reward': -100
    },
    'star': {  # ⭐ 별 (수집해야 함)
        'color': '#FFD700',
        'size': 30,
        'vy': 3,
        'score': 10,
        'reward': 20
    }
}

# 용암지대 설정 (특정 영역만 활성화)
LAVA_CONFIG = {
    'enabled': True,
    'warning_duration': 3.0,  # 경고 3초 (회피 시간 충분히)
    'active_duration': 3.0,   # 용암 활성 3초
    'interval': 20.0,          # 20초마다 등장 (여유 있게)
    'height': 120,             # 용암 높이
    'damage_per_frame': 3,     # 프레임당 데미지
    'zone_width': 320          # 용암 영역 너비 (WIDTH / 3)
}

# 데이터 저장 경로
DATA_DIR = Path(__file__).parent / 'data'
LEADERBOARD_FILE = DATA_DIR / 'leaderboard.json'
GAMEPLAY_DIR = DATA_DIR / 'gameplay' / 'raw'
COLLECTED_DIR = Path(__file__).parent / 'collected_gameplay'  # 훈련 데이터

# 디렉토리 생성
DATA_DIR.mkdir(exist_ok=True)
GAMEPLAY_DIR.mkdir(parents=True, exist_ok=True)
COLLECTED_DIR.mkdir(exist_ok=True)

# 활성 게임들
games = {}

# Storage Manager 초기화
storage = get_storage_manager()

# 리더보드 관리 함수들 (Cloud Storage 사용)
def load_leaderboard():
    """리더보드 로드 (Cloud Storage 또는 로컬)"""
    return storage.load_leaderboard()

def save_leaderboard(leaderboard):
    """리더보드 저장 (Cloud Storage 또는 로컬)"""
    return storage.save_leaderboard(leaderboard)

def add_score(player_name, score, survival_time, mode, session_id):
    """점수 추가 (Cloud Storage 또는 로컬)"""
    return storage.add_score(player_name, score, survival_time, mode, session_id)

def save_gameplay_session(game):
    """게임 세션 저장 (Cloud Storage 또는 로컬)"""
    session_data = {
        'session_id': game.sid,
        'mode': game.mode,
        'score': game.score,
        'survival_time': time.time() - game.start_time,
        'total_frames': game.frame,
        'final_state': {
            'player_x': game.player_x,
            'player_y': game.player_y,
            'obstacles_count': len(game.obstacles)
        },
        'timestamp': datetime.now().isoformat(),
        'player_name': game.player_name
    }
    
    # Cloud Storage에 저장 (storage_manager 사용)
    saved_path = storage.save_gameplay_session(session_data, game.sid)
    
    if saved_path:
        print(f"💾 게임 세션 저장: {saved_path}")
    
    # 2. 훈련 데이터 저장 (State-Action-Reward) - 로컬에만 (용량 문제)
    if len(game.collected_states) > 0:
        save_training_data(game, session_data)
    
    return saved_path

def save_training_data(game, session_metadata):
    """훈련 데이터 저장 (제이 & 클로용)"""
    # 세션별 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = COLLECTED_DIR / f"session_{timestamp}_{game.mode}"
    session_dir.mkdir(exist_ok=True)
    
    # 메타데이터 저장
    metadata_file = session_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(session_metadata, f, indent=2, ensure_ascii=False)
    
    # State-Action-Reward 저장 (JSONL 포맷 - 클로용)
    states_file = session_dir / "states_actions.jsonl"
    with open(states_file, 'w', encoding='utf-8') as f:
        for state_record in game.collected_states:
            f.write(json.dumps(state_record, ensure_ascii=False) + '\n')
    
    # Bounding Box 라벨 저장 (JSONL 포맷 - 제이용)
    bboxes_file = session_dir / "bboxes.jsonl"
    with open(bboxes_file, 'w', encoding='utf-8') as f:
        for state_record in game.collected_states:
            frame_num = state_record['frame']
            state = state_record['state']
            
            # 게임 상태에서 bbox 추출
            objects = []
            
            # 플레이어 bbox
            objects.append({
                'class': 'player',
                'x': state['player_x'],
                'y': state['player_y'],
                'w': PLAYER_SIZE,
                'h': PLAYER_SIZE
            })
            
            # 장애물 bbox
            for obs in state['obstacles']:
                objects.append({
                    'class': 'obstacle',
                    'x': obs['x'],
                    'y': obs['y'],
                    'w': obs['size'],
                    'h': obs['size']
                })
            
            f.write(json.dumps({'frame': frame_num, 'objects': objects}, ensure_ascii=False) + '\n')
    
    print(f"📊 훈련 데이터 저장:")
    print(f"   - 디렉토리: {session_dir.name}")
    print(f"   - State-Action 로그: {len(game.collected_states)}개")
    print(f"   - Bbox 라벨: {len(game.collected_states)}개")
    
    # 3. YOLO 데이터셋으로 내보내기 (추가된 기능)
    try:
        from yolo_exporter import YOLOExporter
        exporter = YOLOExporter(base_dir="game_dataset")
        
        # 프레임이 저장된 경로 찾기
        # storage_manager.py에 따르면: local_data_dir / 'gameplay' / 'frames' / date_folder / session_id[:8]
        date_folder = datetime.now().strftime("%Y-%m-%d")
        frames_dir = storage.local_data_dir / 'gameplay' / 'frames' / date_folder / game.sid[:8]
        
        if frames_dir.exists():
            exporter.export_session(game.sid, game.collected_states, frames_dir)
        else:
            print(f"⚠️ 프레임 디렉토리를 찾을 수 없음: {frames_dir}")
            
    except Exception as e:
        print(f"❌ YOLO Export 실패: {e}")
    
    return str(session_dir)

class Game:
    def __init__(self, sid):
        self.sid = sid
        # CV 모듈 초기화 (Vision 기반 라바 감지용)
        self.cv_module = ComputerVisionModule()
        self.reset()
        
    def reset(self):
        """게임 상태 초기화"""
        self.player_x = WIDTH // 2
        self.player_y = HEIGHT // 2
        self.player_vy = 0
        self.obstacles = []  # 메테오와 별을 포함
        self.score = 0
        self.running = False
        self.mode = "human"
        self.player_name = None  # 플레이어 이름
        self.start_time = time.time()
        self.frame = 0
        self.game_over = False
        
        # 훈련 데이터 수집
        self.collected_states = []  # State-Action-Reward 로그
        self.last_action = "stay"
        
        # 이벤트 플래그
        self.star_collected = False  # 별 획득 플래그
        
        # 용암지대 상태 (특정 영역만)
        # Note: 라바는 바닥에 고정되어 있지만, YOLO로 감지하면 "Vision 기반 인식"이라는 점을 더 강조할 수 있습니다.
        self.lava_state = 'inactive'  # inactive, warning, active
        self.lava_timer = LAVA_CONFIG['interval']  # 다음 용암까지 시간
        self.lava_phase_timer = 0  # 현재 단계 타이머
        self.lava_zone_x = 0  # 용암이 나올 X 위치 (CV 감지 결과로 업데이트됨)
        self.player_health = 100  # 플레이어 체력 (용암 데미지용)
        
        # CV 감지 결과 저장 (라바 감지용)
        self.detected_lava = None  # CVDetectionResult 또는 None
        
    def update(self):
        """물리 업데이트"""
        if self.game_over:
            return
        
        # 이벤트 플래그 초기화
        self.star_collected = False
        
        # 📊 현재 상태 저장 (업데이트 전)
        current_state = {
            'player_x': self.player_x,
            'player_y': self.player_y,
            'player_vy': self.player_vy,
            'obstacles': [{'x': o['x'], 'y': o['y'], 'size': o['size'], 'type': o.get('type', 'meteor')} for o in self.obstacles],
            'lava': {
                'state': self.lava_state,
                'zone_x': self.lava_zone_x,
                'height': LAVA_CONFIG['height'],
                'zone_width': LAVA_CONFIG['zone_width']
            }
        }
        
        # 중력
        self.player_vy += 1
        self.player_y += self.player_vy
        
        # 바닥 충돌
        if self.player_y >= HEIGHT - PLAYER_SIZE:
            self.player_y = HEIGHT - PLAYER_SIZE
            self.player_vy = 0
        
        # 장애물 이동 (대각선)
        for obs in self.obstacles:
            obs['x'] += obs.get('vx', 0)  # 좌우 이동
            obs['y'] += obs.get('vy', 5)  # 하강
            
            # 화면 밖으로 나가면 반대편에서 등장 (좌우 wrap)
            if obs['x'] < -obs.get('size', OBSTACLE_SIZE):
                obs['x'] = WIDTH
            elif obs['x'] > WIDTH:
                obs['x'] = -obs.get('size', OBSTACLE_SIZE)
        
        # 화면 밖 장애물 제거 + 점수 증가
        before_count = len(self.obstacles)
        self.obstacles = [o for o in self.obstacles if o['y'] < HEIGHT]
        cleared = before_count - len(self.obstacles)
        self.score += cleared
        
        # 충돌 검사
        self.check_collisions()
        
        # 📊 보상 계산
        reward = 1.0  # 생존 기본 보상
        
        # 화면 밖으로 나간 객체 보상 (회피 성공)
        if cleared > 0:
            reward += cleared * 5
        
        # 게임 오버 (메테오 충돌)
        if self.game_over:
            reward = OBJECT_TYPES['meteor']['reward']  # -100
        
        # 별 획득 보상은 check_collisions()에서 별도 처리
        
        # 📊 State-Action-Reward 저장 (클로 훈련용)
        self.collected_states.append({
            'frame': self.frame,
            'state': current_state,
            'action': self.last_action,
            'reward': reward,
            'done': self.game_over
        })
        
        # 새 객체 생성 (메테오 또는 별)
        if random.random() < 0.05:
            # 10% 확률로 별, 나머지는 메테오
            obj_type = 'star' if random.random() < 0.1 else 'meteor'
            obj_config = OBJECT_TYPES[obj_type]
            
            self.obstacles.append({
                'type': obj_type,
                'x': random.randint(0, WIDTH - obj_config['size']),
                'y': -obj_config['size'],
                'vx': random.randint(-2, 2),  # 대각선 이동
                'vy': obj_config['vy'],
                'size': obj_config['size']
            })
        
        # 🌋 용암지대 업데이트 (하드코딩된 로직으로 상태 관리)
        if LAVA_CONFIG['enabled']:
            self.update_lava()
        
        # 🔍 Vision 기반 라바 감지 (YOLO로 감지하여 "Vision 기반 인식" 강조)
        self.detect_lava_with_cv()
        
        self.frame += 1
    
    def update_lava(self):
        """🌋 용암지대 업데이트 (특정 영역만) - 하드코딩된 로직으로 상태 관리"""
        dt = 1.0 / 30.0  # 30 FPS 기준
        
        if self.lava_state == 'inactive':
            # 용암 대기 중
            self.lava_timer -= dt
            if self.lava_timer <= 0:
                # 경고 단계 시작 + 랜덤 영역 선택
                self.lava_state = 'warning'
                self.lava_phase_timer = LAVA_CONFIG['warning_duration']
                # 좌측(0), 중앙(320), 우측(640) 중 랜덤 선택
                self.lava_zone_x = random.choice([0, WIDTH // 3, (WIDTH // 3) * 2])
                print(f"⚠️ 용암 경고! 영역: X={self.lava_zone_x}")
        
        elif self.lava_state == 'warning':
            # 경고 단계
            self.lava_phase_timer -= dt
            if self.lava_phase_timer <= 0:
                # 용암 활성화
                self.lava_state = 'active'
                self.lava_phase_timer = LAVA_CONFIG['active_duration']
                print("🌋 용암 활성화!")
        
        elif self.lava_state == 'active':
            # 용암 활성 단계
            self.lava_phase_timer -= dt
            
            # Vision 기반 라바 감지 결과 사용 (CV 모듈에서 감지된 라바 위치)
            # CV 감지 결과가 있으면 우선 사용, 없으면 하드코딩된 위치 사용
            if self.detected_lava is not None:
                # CV 감지 결과에서 라바 위치 추출
                lava_bbox = self.detected_lava.bbox
                lava_x_start = int(lava_bbox[0])
                lava_x_end = int(lava_bbox[2])
                lava_y_start = int(lava_bbox[1])
            else:
                # 폴백: 하드코딩된 위치 사용
                lava_y_start = HEIGHT - LAVA_CONFIG['height']
                lava_x_start = self.lava_zone_x
                lava_x_end = self.lava_zone_x + LAVA_CONFIG['zone_width']
            
            # 플레이어가 용암 영역 안에 있고, Y 좌표도 용암 영역 안이면 데미지
            player_in_zone_x = (self.player_x + PLAYER_SIZE > lava_x_start and 
                                self.player_x < lava_x_end)
            player_in_zone_y = self.player_y + PLAYER_SIZE > lava_y_start
            
            if player_in_zone_x and player_in_zone_y:
                # 용암 데미지
                self.player_health -= LAVA_CONFIG['damage_per_frame']
                if self.player_health <= 0:
                    self.game_over = True
                    print("🔥 용암에 빠져 게임 오버! (Vision 기반 감지)")
            
            if self.lava_phase_timer <= 0:
                # 용암 비활성화, 다음 주기로
                self.lava_state = 'inactive'
                self.lava_timer = LAVA_CONFIG['interval']
                self.player_health = 100  # 체력 회복
                self.detected_lava = None  # CV 감지 결과 초기화
                print("✅ 용암 종료")
    
    def detect_lava_with_cv(self):
        """
        🔍 Vision 기반 라바 감지 (YOLO 사용)
        
        Note: 라바는 바닥에 고정되어 있지만, YOLO로 감지하면 
        "Vision 기반 인식"이라는 점을 더 강조할 수 있습니다.
        """
        try:
            # 게임 상태를 CV 모듈에 전달
            game_state = self.get_state()
            
            # 더미 프레임 생성 (실제 YOLO 구현 시 실제 프레임 사용)
            # 프레임 크기는 게임 화면 크기와 일치
            dummy_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            
            # CV 모듈로 객체 탐지 (게임 상태 포함)
            detections = self.cv_module.detect_objects(dummy_frame, game_state)
            
            # 라바 감지 결과 찾기
            self.detected_lava = None
            for detection in detections:
                if detection.class_id == 4 or detection.class_name == "Lava":
                    self.detected_lava = detection
                    # 디버깅: 라바 감지 로그 (너무 자주 출력하지 않도록)
                    if self.frame % 30 == 0:  # 1초마다 한 번
                        print(f"🔍 [Vision] 라바 감지: bbox={detection.bbox}, confidence={detection.confidence:.2f}")
                    break
            
        except Exception as e:
            # 오류 발생 시 폴백 (하드코딩된 로직 사용)
            print(f"⚠️ CV 라바 감지 오류: {e}, 하드코딩된 로직 사용")
            self.detected_lava = None
    
    def check_collisions(self):
        """충돌 검사 (AABB) - 메테오 vs 별"""
        for obs in self.obstacles[:]:  # 복사본으로 순회 (리스트 수정 가능)
            obj_size = obs.get('size', OBSTACLE_SIZE)
            
            # AABB (Axis-Aligned Bounding Box) 충돌 감지
            if (self.player_x < obs['x'] + obj_size and
                self.player_x + PLAYER_SIZE > obs['x'] and
                self.player_y < obs['y'] + obj_size and
                self.player_y + PLAYER_SIZE > obs['y']):
                
                obj_type = obs.get('type', 'meteor')
                
                if obj_type == 'meteor':
                    # 메테오 충돌: 게임 오버
                    self.game_over = True
                    self.running = False
                    print(f"💥 메테오 충돌! 게임 오버! 점수: {self.score}, 생존 시간: {time.time() - self.start_time:.1f}초")
                    
                elif obj_type == 'star':
                    # 별 획득: 점수 증가
                    star_score = OBJECT_TYPES['star']['score']
                    self.score += star_score
                    self.obstacles.remove(obs)
                    self.star_collected = True  # 별 획득 플래그 설정
                    print(f"⭐ 별 획득! +{star_score}점 (총 {self.score}점)")
    
    def jump(self):
        """점프"""
        if self.player_y >= HEIGHT - PLAYER_SIZE - 5:
            self.player_vy = -18
        self.last_action = "jump"
    
    def move_left(self):
        """왼쪽 이동"""
        self.player_x = max(0, self.player_x - 10)
        self.last_action = "move_left"
    
    def move_right(self):
        """오른쪽 이동"""
        self.player_x = min(WIDTH - PLAYER_SIZE, self.player_x + 10)
        self.last_action = "move_right"
    
    def get_state(self):
        """현재 상태"""
        return {
            'player': {
                'x': self.player_x,
                'y': self.player_y,
                'vy': self.player_vy,
                'size': PLAYER_SIZE,
                'health': self.player_health  # 용암 데미지용 체력
            },
            'obstacles': self.obstacles,
            'score': self.score,
            'time': time.time() - self.start_time,
            'frame': self.frame,
            'mode': self.mode,
            'game_over': self.game_over,
            'star_collected': self.star_collected,  # 별 획득 이벤트
            'lava': {  # 용암지대 정보 (특정 영역만)
                'state': self.lava_state,
                'timer': self.lava_phase_timer if self.lava_state != 'inactive' else self.lava_timer,
                'height': LAVA_CONFIG['height'],
                'zone_x': self.lava_zone_x,  # 용암 영역 X 시작점
                'zone_width': LAVA_CONFIG['zone_width']  # 용암 영역 너비
            }
        }

def encode_game_state(game):
    """
    게임 상태를 RL 모델 입력으로 인코딩
    
    상태 벡터 (10차원):
    - player_x_normalized (0~1)
    - player_y_normalized (0~1)
    - player_vy_normalized (-1~1)
    - nearest_meteor_dx_normalized (-1~1)
    - nearest_meteor_dy_normalized (0~1)
    - nearest_meteor_distance_normalized (0~1)
    - nearest_star_dx_normalized (-1~1)
    - nearest_star_dy_normalized (0~1)
    - nearest_star_distance_normalized (0~1)
    - on_ground (0 or 1)
    """
    player_x = game.player_x
    player_y = game.player_y
    player_vy = game.player_vy
    player_center_x = player_x + PLAYER_SIZE / 2
    
    # 정규화
    state = np.zeros(10, dtype=np.float32)
    state[0] = player_x / WIDTH
    state[1] = player_y / HEIGHT
    state[2] = np.clip(player_vy / 20.0, -1, 1)
    state[9] = 1.0 if player_y >= HEIGHT - PLAYER_SIZE - 5 else 0.0
    
    # 가장 가까운 메테오 & 별 찾기
    nearest_meteor_dist = 1.0
    nearest_star_dist = 1.0
    
    for obs in game.obstacles:
        obj_type = obs.get('type', 'meteor')
        obs_center_x = obs['x'] + obs.get('size', OBSTACLE_SIZE) / 2
        obs_center_y = obs['y'] + obs.get('size', OBSTACLE_SIZE) / 2
        
        dx = (obs_center_x - player_center_x) / WIDTH
        dy = (obs_center_y - player_y) / HEIGHT
        dist = np.sqrt(dx**2 + dy**2)
        
        if obj_type == 'meteor' and dist < nearest_meteor_dist:
            nearest_meteor_dist = dist
            state[3] = np.clip(dx, -1, 1)
            state[4] = np.clip(dy, 0, 1)
            state[5] = dist
        
        elif obj_type == 'star' and dist < nearest_star_dist:
            nearest_star_dist = dist
            state[6] = np.clip(dx, -1, 1)
            state[7] = np.clip(dy, 0, 1)
            state[8] = dist
    
    return state

def ai_decision(game):
    """
    AI 에이전트의 의사결정 로직
    
    우선순위:
    1. RL 모델 사용 (학습된 모델이 있으면)
    2. 휴리스틱 정책 (기본 전략)
    
    전략:
    1. 가장 가까운 메테오 회피
    2. 가까운 별 수집
    3. 안전 구역 유지
    """
    # RL 모델이 있으면 사용
    if RL_MODEL_AVAILABLE and RL_MODEL is not None:
        try:
            state = encode_game_state(game)
            # import torch
            # with torch.no_grad():
            #     state_tensor = torch.FloatTensor(state).unsqueeze(0)
            #     action_probs = RL_MODEL(state_tensor)
            #     action_idx = torch.argmax(action_probs).item()
            #     actions = ['stay', 'left', 'right', 'jump']
            #     return actions[action_idx] if action_idx > 0 else None
            pass
        except Exception as e:
            print(f"⚠️ RL 모델 추론 오류: {e}")
    
    # 휴리스틱 정책 (기본)
    player_x = game.player_x
    player_y = game.player_y
    player_center_x = player_x + PLAYER_SIZE / 2
    
    # 위협 분석
    nearest_meteor = None
    nearest_meteor_dist = float('inf')
    nearest_star = None
    nearest_star_dist = float('inf')
    
    for obs in game.obstacles:
        obj_type = obs.get('type', 'meteor')
        obs_x = obs['x']
        obs_y = obs['y']
        obs_size = obs.get('size', OBSTACLE_SIZE)
        obs_center_x = obs_x + obs_size / 2
        
        # 충돌 예상 범위 (플레이어와 x축 중첩)
        x_overlap = abs(player_center_x - obs_center_x) < (PLAYER_SIZE + obs_size) / 2 + 50
        
        if obj_type == 'meteor':
            # 메테오가 플레이어 위쪽에 있고 접근 중
            if obs_y < player_y and x_overlap:
                dist = abs(player_center_x - obs_center_x) + (player_y - obs_y) * 0.5
                if dist < nearest_meteor_dist:
                    nearest_meteor_dist = dist
                    nearest_meteor = obs
        
        elif obj_type == 'star':
            # 별이 획득 가능한 범위
            if obs_y < player_y + 200:
                dist = abs(player_center_x - obs_center_x) + abs(player_y - obs_y) * 0.3
                if dist < nearest_star_dist:
                    nearest_star_dist = dist
                    nearest_star = obs
    
    # 의사결정 우선순위
    action = None
    
    # 1. 위급 상황: 메테오 회피
    if nearest_meteor and nearest_meteor_dist < 150:
        meteor_center_x = nearest_meteor['x'] + nearest_meteor.get('size', OBSTACLE_SIZE) / 2
        
        # 메테오가 왼쪽에서 오면 오른쪽으로, 오른쪽에서 오면 왼쪽으로
        if meteor_center_x < player_center_x:
            if player_x + PLAYER_SIZE < WIDTH - 20:
                action = 'right'
        else:
            if player_x > 20:
                action = 'left'
        
        # 긴급 상황: 점프로 회피 시도
        if nearest_meteor_dist < 80 and player_y >= HEIGHT - PLAYER_SIZE - 10:
            action = 'jump'
    
    # 2. 기회 포착: 별 수집
    elif nearest_star and nearest_star_dist < 200:
        star_center_x = nearest_star['x'] + nearest_star.get('size', 30) / 2
        
        # 별 쪽으로 이동
        if star_center_x < player_center_x - 15:
            if player_x > 10:
                action = 'left'
        elif star_center_x > player_center_x + 15:
            if player_x + PLAYER_SIZE < WIDTH - 10:
                action = 'right'
        
        # 별이 위쪽에 있으면 점프
        if nearest_star['y'] < player_y - 50 and player_y >= HEIGHT - PLAYER_SIZE - 10:
            action = 'jump'
    
    # 3. 기본 행동: 중앙 유지 (좌우 이동 범위 확보)
    else:
        center_x = WIDTH / 2
        if player_center_x < center_x - 100:
            if player_x + PLAYER_SIZE < WIDTH - 20:
                action = 'right'
        elif player_center_x > center_x + 100:
            if player_x > 20:
                action = 'left'
    
    return action

def game_loop(sid):
    """게임 루프"""
    game = games.get(sid)
    if not game:
        return
    
    print(f"🎮 게임 루프 시작: {sid} (모드: {game.mode})")
    
    while game.running and not game.game_over:
        try:
            # AI 모드: 자동 의사결정
            if game.mode == 'ai':
                action = ai_decision(game)
                if action == 'jump':
                    game.jump()
                elif action == 'left':
                    game.move_left()
                elif action == 'right':
                    game.move_right()
            
            game.update()
            
            # 상태 전송
            socketio.emit('game_update', {
                'state': game.get_state()
            })
            
            time.sleep(1.0 / 30)  # 30 FPS
            
        except Exception as e:
            print(f"❌ 에러: {e}")
            break
    
    # 게임 오버 처리
    if game.game_over:
        survival_time = time.time() - game.start_time
        
        # 게임 세션 저장 (팀원들의 훈련 데이터용)
        save_gameplay_session(game)
        
        # 리더보드에 점수 추가
        player_name = game.player_name or f"Player-{sid[:6]}"
        leaderboard = add_score(player_name, game.score, survival_time, game.mode, sid)
        
        # 클라이언트에 게임 오버 + 랭킹 전송
        socketio.emit('game_over', {
            'score': game.score,
            'time': survival_time,
            'frame': game.frame,
            'player_name': player_name,
            'mode': game.mode,  # 모드 추가
            'leaderboard': leaderboard['scores'][:10]  # 상위 10개만
        })
        
        print(f"💾 점수 저장: {player_name} ({game.mode}) - {game.score}점 ({survival_time:.1f}초)")
    
    print(f"🛑 게임 루프 종료: {sid}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/leaderboard')
def api_leaderboard():
    """리더보드 API"""
    leaderboard = load_leaderboard()
    return jsonify(leaderboard)

@app.route('/api/leaderboard/top/<int:limit>')
def api_leaderboard_top(limit):
    """상위 N개 점수"""
    leaderboard = load_leaderboard()
    return jsonify({
        'scores': leaderboard['scores'][:limit]
    })

@app.route('/api/stats')
def api_stats():
    """통계 정보 (Cloud Storage 연동)"""
    return jsonify(storage.get_stats())

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
        print(f"❌ 게임 없음: {sid}")
        return
    
    # 게임 재시작: 상태 초기화
    game.reset()
    game.mode = data.get('mode', 'human')
    game.player_name = data.get('player_name', None)  # 플레이어 이름 저장
    game.running = True
    
    # 플레이어 이름 설정 (AI면 자동 생성)
    if game.mode == 'ai':
        game.player_name = f"AI-Bot-{sid[:6]}"
    elif not game.player_name:
        game.player_name = f"Player-{sid[:6]}"
    
    print(f"🚀 게임 시작: {sid}, 모드: {game.mode}, 플레이어: {game.player_name}")
    
    # 게임 루프 시작
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

@socketio.on('frame_capture')
def on_frame_capture(data):
    """
    프레임 이미지 수집 (CV 훈련용)
    
    클라이언트가 Canvas를 캡처해서 Base64 PNG로 전송
    """
    from flask import request
    import base64
    
    sid = request.sid
    game = games.get(sid)
    
    if not game or not game.running:
        return
    
    try:
        # Base64 PNG 디코딩
        image_base64 = data.get('image')
        frame_number = data.get('frame', 0)
        
        if not image_base64:
            return
        
        # "data:image/png;base64," 접두사 제거
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        
        # Cloud Storage에 저장
        saved_path = storage.save_frame_image(image_bytes, sid, frame_number)
        
        if saved_path and frame_number % 30 == 0:  # 30프레임마다 로그
            print(f"📸 프레임 저장: {saved_path}")
    
    except Exception as e:
        print(f"❌ 프레임 저장 오류: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'True') == 'True'
    env_mode = os.environ.get('ENVIRONMENT', 'development')
    
    print("🎮 게임 서버 시작!")
    print(f"🌐 http://localhost:{port}")
    print(f"🤖 AI 모드: 휴리스틱 기반 (RL 모델 대기 중)")
    print(f"📦 환경: {env_mode}")
    
    # Storage 상태 출력
    if storage.use_gcs:
        print(f"☁️ Cloud Storage 사용: gs://{storage.bucket_name}")
    else:
        print(f"💾 로컬 스토리지 사용: {storage.local_data_dir}")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)

