"""
AI Module - Reinforcement Learning Policy

Chloe Lee (cl4490) 담당 모듈
PPO/DQN 기반 게임 AI 정책

난이도 레벨 시스템:
- Level 1 (Easy): 간단한 휴리스틱 (기본 회피만)
- Level 2 (Medium): PPO 모델 기반 (Vision → State → Policy)
- Level 3 (Hard): 고급 휴리스틱 + PPO 앙상블
- Level 4 (Expert): 풀 앙상블 (PPO + DQN + 휴리스틱)

수정: 2025-11-29
- Level 2에 실제 PPO 모델 통합
- state_encoder.py 사용
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import random
from pathlib import Path

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 없음 - 휴리스틱 모드만 사용 가능")

# State Encoder (우리가 만든 모듈)
try:
    from .state_encoder import encode_state, game_state_to_detections, STATE_DIM, ACTION_LIST
except ImportError:
    try:
        from state_encoder import encode_state, game_state_to_detections, STATE_DIM, ACTION_LIST
    except ImportError:
        print("⚠️ state_encoder.py 없음 - 휴리스틱 모드만 사용 가능")
        STATE_DIM = 26
        ACTION_LIST = ["stay", "left", "right", "jump"]
        encode_state = None
        game_state_to_detections = None


# ============================================================================
# Policy Network (PPO용)
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Actor Network: State → Action Probabilities
    
    Architecture: 26 → 256 → 256 → 128 → 4
    """
    
    def __init__(self, state_dim=26, action_dim=4, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
        # Xavier 초기화
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """
    Critic Network: State → Value Estimate
    """
    
    def __init__(self, state_dim=26, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# ============================================================================
# AI Strategy Classes
# ============================================================================

class AIStrategy:
    """AI 전략 베이스 클래스"""
    
    def __init__(self, level: int, name: str):
        self.level = level
        self.name = name
    
    def make_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        raise NotImplementedError


class Level1Strategy(AIStrategy):
    """
    Level 1 (Easy) - 간단한 휴리스틱
    
    전략:
    - 기본적인 메테오 회피만
    - 별은 무시
    """
    
    def __init__(self):
        super().__init__(level=1, name="Easy")
        self.DETECTION_RANGE = 200
        self.DANGER_RANGE = 100
    
    def make_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        player = game_state.get('player', {})
        obstacles = game_state.get('obstacles', [])
        
        player_x = player.get('x', 480)
        player_y = player.get('y', 360)
        player_size = player.get('size', 50)
        player_center_x = player_x + player_size / 2
        
        # 가장 가까운 메테오 찾기
        nearest_meteor = None
        nearest_dist = float('inf')
        
        for obs in obstacles:
            if obs.get('type') != 'meteor':
                continue
            
            obs_x = obs.get('x', 0)
            obs_y = obs.get('y', 0)
            obs_size = obs.get('size', 50)
            obs_center_x = obs_x + obs_size / 2
            
            if obs_y < player_y:
                x_overlap = abs(player_center_x - obs_center_x) < self.DETECTION_RANGE
                if x_overlap:
                    dist = abs(player_center_x - obs_center_x) + (player_y - obs_y) * 0.5
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_meteor = obs
        
        # 메테오 회피
        if nearest_meteor and nearest_dist < self.DANGER_RANGE:
            meteor_center_x = nearest_meteor['x'] + nearest_meteor.get('size', 50) / 2
            if meteor_center_x < player_center_x:
                return 'right'
            else:
                return 'left'
        
        return None


class Level2Strategy(AIStrategy):
    """
    Level 2 (Medium) - PPO 모델 기반
    
    전략:
    - 학습된 PPO 모델 사용
    - state_encoder로 게임 상태 → 26-dim 벡터 변환
    - 모델 없으면 휴리스틱 폴백
    """
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(level=2, name="Medium (PPO)")
        self.model_path = model_path
        self.policy_net = None
        self.device = None
        self.fallback_strategy = Level1Strategy()
        
        # PPO 모델 로드
        self._load_ppo_model()
    
    def _load_ppo_model(self):
        """PPO 모델 로드"""
        if not TORCH_AVAILABLE:
            print("⚠️ Level 2: PyTorch 없음, 휴리스틱으로 폴백")
            return
        
        if not self.model_path:
            print("⚠️ Level 2: 모델 경로 없음, 휴리스틱으로 폴백")
            return
        
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                print(f"⚠️ Level 2: 모델 파일 없음 ({self.model_path})")
                return
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Checkpoint 로드
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # State/Action 차원 확인
            state_dim = checkpoint.get('state_dim', STATE_DIM)
            action_dim = checkpoint.get('action_dim', len(ACTION_LIST))
            
            # Policy Network 생성 및 가중치 로드
            self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
            
            if 'policy_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            elif 'policy' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['policy'])
            else:
                # 직접 state_dict인 경우
                self.policy_net.load_state_dict(checkpoint)
            
            self.policy_net.eval()
            print(f"✅ Level 2: PPO 모델 로드 성공 ({self.model_path})")
            print(f"   State dim: {state_dim}, Action dim: {action_dim}")
            
        except Exception as e:
            print(f"⚠️ Level 2: PPO 모델 로드 실패 ({e})")
            self.policy_net = None
    
    def make_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """PPO 모델 기반 의사결정"""
        # PPO 모델이 있으면 사용
        if self.policy_net is not None and encode_state is not None:
            try:
                return self._ppo_decision(game_state)
            except Exception as e:
                print(f"⚠️ Level 2: PPO 추론 오류 ({e})")
        
        # 폴백: Level 1 전략 사용
        return self.fallback_strategy.make_decision(game_state)
    
    def _ppo_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """
        PPO 모델 기반 의사결정
        
        1. game_state → detections 변환
        2. detections → 26-dim state vector
        3. PPO 추론 → action probabilities
        4. argmax → action
        """
        # Step 1: game_state → detections 변환
        detections = game_state_to_detections(game_state)
        
        # Step 2: encode_state()로 26-dim 벡터 생성
        state_vec = encode_state(detections, game_state)
        
        # Step 3: PPO 추론
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            action_probs = self.policy_net(state_tensor)
            action_idx = torch.argmax(action_probs, dim=-1).item()
        
        # Step 4: action index → action string
        action = ACTION_LIST[action_idx]
        
        # 'stay'는 None으로 반환 (app.py 호환)
        if action == 'stay':
            return None
        
        return action


class Level3Strategy(AIStrategy):
    """
    Level 3 (Hard) - 고급 휴리스틱 + PPO
    
    전략:
    - PPO 모델 + 휴리스틱 보완
    - 용암 회피 로직 추가
    - 별 수집 전략
    """
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(level=3, name="Hard")
        self.ppo_strategy = Level2Strategy(model_path=model_path)
        self.METEOR_DANGER_RANGE = 150
        self.STAR_COLLECT_RANGE = 200
        self.EMERGENCY_RANGE = 80
    
    def make_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """PPO + 휴리스틱 앙상블"""
        player = game_state.get('player', {})
        obstacles = game_state.get('obstacles', [])
        lava = game_state.get('lava', {})
        
        player_x = player.get('x', 480)
        player_y = player.get('y', 360)
        player_size = player.get('size', 50)
        player_center_x = player_x + player_size / 2
        
        WIDTH = 960
        HEIGHT = 720
        
        # 긴급 상황 체크: 메테오가 매우 가까움
        for obs in obstacles:
            if obs.get('type') != 'meteor':
                continue
            obs_x = obs.get('x', 0)
            obs_y = obs.get('y', 0)
            obs_size = obs.get('size', 50)
            obs_center_x = obs_x + obs_size / 2
            
            dist = np.sqrt((player_center_x - obs_center_x)**2 + (player_y - obs_y)**2)
            if dist < self.EMERGENCY_RANGE and obs_y < player_y:
                # 긴급 회피
                if player_y >= HEIGHT - player_size - 10:
                    return 'jump'
        
        # 용암 회피 (최우선)
        if lava.get('state') == 'active':
            lava_zone_x = lava.get('zone_x', 0)
            lava_zone_width = lava.get('zone_width', 320)
            lava_zone_end = lava_zone_x + lava_zone_width
            
            if player_x + player_size > lava_zone_x and player_x < lava_zone_end:
                if player_center_x < WIDTH / 2:
                    return 'left'
                else:
                    return 'right'
        
        # PPO 모델 의사결정
        ppo_action = self.ppo_strategy.make_decision(game_state)
        if ppo_action:
            return ppo_action
        
        # 용암 경고 회피
        if lava.get('state') == 'warning':
            lava_zone_x = lava.get('zone_x', 0)
            lava_zone_width = lava.get('zone_width', 320)
            lava_zone_end = lava_zone_x + lava_zone_width
            
            if player_x + player_size > lava_zone_x - 50 and player_x < lava_zone_end + 50:
                if player_center_x < WIDTH / 2:
                    return 'left'
                else:
                    return 'right'
        
        return None


class Level4Strategy(AIStrategy):
    """
    Level 4 (Expert) - 풀 앙상블
    
    전략:
    - PPO + 휴리스틱 + 용암/별 전략
    - 모든 요소 고려
    """
    
    def __init__(self, ppo_model_path: Optional[str] = None, dqn_model_path: Optional[str] = None):
        super().__init__(level=4, name="Expert")
        self.level3_strategy = Level3Strategy(model_path=ppo_model_path)
    
    def make_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """Level 3 전략 + 추가 최적화"""
        return self.level3_strategy.make_decision(game_state)


# ============================================================================
# AI Level Manager
# ============================================================================

class AILevelManager:
    """AI 난이도 레벨 관리자"""
    
    def __init__(self, ppo_model_path: Optional[str] = None, dqn_model_path: Optional[str] = None):
        """
        초기화
        
        Args:
            ppo_model_path: PPO 모델 경로 (Level 2, 3, 4에서 사용)
            dqn_model_path: DQN 모델 경로 (선택적)
        """
        self.ppo_model_path = ppo_model_path
        self.dqn_model_path = dqn_model_path
        
        self.strategies = {
            1: Level1Strategy(),
            2: Level2Strategy(model_path=ppo_model_path),
            3: Level3Strategy(model_path=ppo_model_path),
            4: Level4Strategy(ppo_model_path=ppo_model_path, dqn_model_path=dqn_model_path)
        }
        self.current_level = 2  # 기본값: Level 2 (PPO)
        
        print(f"🤖 AI Level Manager 초기화")
        print(f"   - Level 1: Easy (휴리스틱)")
        print(f"   - Level 2: Medium (PPO)")
        print(f"   - Level 3: Hard (PPO + 휴리스틱)")
        print(f"   - Level 4: Expert (앙상블)")
    
    def set_level(self, level: int):
        """난이도 레벨 설정"""
        if level not in self.strategies:
            print(f"⚠️ Invalid level: {level}. Using default (2).")
            level = 2
        self.current_level = level
    
    def make_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """현재 레벨의 전략으로 의사결정"""
        strategy = self.strategies[self.current_level]
        return strategy.make_decision(game_state)
    
    def get_level_info(self) -> Dict[str, Any]:
        """현재 레벨 정보 반환"""
        strategy = self.strategies[self.current_level]
        return {
            'level': self.current_level,
            'name': strategy.name,
            'description': f"Level {self.current_level}: {strategy.name}"
        }


# ============================================================================
# Legacy Support (AIModule class)
# ============================================================================

class AIDecisionResult:
    """AI 의사결정 결과 (레거시 호환)"""
    
    def __init__(self, action: str, confidence: float = 0.5, reasoning: str = ""):
        self.action = action
        self.confidence = confidence
        self.reasoning = reasoning
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp
        }


class AIModule:
    """
    AI 모듈 (레거시 호환용)
    
    새 코드에서는 AILevelManager 사용 권장
    """
    
    def __init__(self, model_path: Optional[str] = None, algorithm: str = "PPO"):
        self.level_manager = AILevelManager(ppo_model_path=model_path)
        self.level_manager.set_level(2)  # Level 2 (PPO) 사용
    
    def make_decision(self, game_state: Dict[str, Any]) -> AIDecisionResult:
        action = self.level_manager.make_decision(game_state)
        if action is None:
            action = 'stay'
        return AIDecisionResult(action=action)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # 테스트 게임 상태
    test_state = {
        'player': {
            'x': 480,
            'y': 670,
            'vy': 0,
            'size': 50,
            'health': 100
        },
        'obstacles': [
            {'type': 'meteor', 'x': 500, 'y': 200, 'size': 50, 'vx': 0, 'vy': 5},
            {'type': 'star', 'x': 300, 'y': 400, 'size': 30, 'vx': 0, 'vy': 3}
        ],
        'lava': {
            'state': 'inactive',
            'zone_x': 0,
            'zone_width': 320,
            'height': 120
        },
        'score': 50,
        'frame': 100
    }
    
    # AI Level Manager 테스트
    ai_manager = AILevelManager(ppo_model_path="models/rl/ppo_agent.pt")
    
    for level in [1, 2, 3, 4]:
        ai_manager.set_level(level)
        action = ai_manager.make_decision(test_state)
        info = ai_manager.get_level_info()
        print(f"Level {level} ({info['name']}): Action = {action}")
