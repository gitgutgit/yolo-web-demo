"""
AI Module - Reinforcement Learning Policy

Chloe Lee (cl4490) 담당 모듈
PPO/DQN 기반 게임 AI 정책

난이도 레벨 시스템:
- Level 1 (Easy): 간단한 휴리스틱 (기본 회피만)
- Level 2 (Medium): 고급 휴리스틱 (회피 + 별 수집 전략)
- Level 3 (Hard): PPO 모델 기반 (없으면 최고급 휴리스틱)

TODO for Chloe:
1. simulate_ai_decision() → real_ppo_decision() 교체
2. 정책 네트워크 훈련 및 로드
3. 실시간 의사결정 최적화
4. 자가 학습 (Self-Play) 구현
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import random
from pathlib import Path

# PyTorch는 선택적 (실제 RL 모델 구현 시 필요)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch (torch) 없음 - 시뮬레이션 모드만 사용 가능")
    # 더미 클래스 (타입 힌트용)
    class nn:
        class Module:
            pass
        class Sequential:
            pass
        class Linear:
            pass
        class ReLU:
            pass
        class Softmax:
            pass

# TODO: Chloe가 추가할 import
# from stable_baselines3 import PPO, DQN
# from ..src.utils.rl_instrumentation import RLInstrumentationLogger


class PolicyNetwork(nn.Module):
    """
    정책 네트워크 (MLP)
    
    Chloe가 구현할 신경망 구조
    """
    
    def __init__(self, state_dim: int = 8, hidden_dim: int = 128, action_dim: int = 4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch (torch)가 필요합니다. 실제 RL 모델 구현 시 사용됩니다.")
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)


class ValueNetwork(nn.Module):
    """
    가치 네트워크 (PPO용)
    
    Chloe가 PPO 구현 시 사용
    """
    
    def __init__(self, state_dim: int = 8, hidden_dim: int = 128):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch (torch)가 필요합니다. 실제 RL 모델 구현 시 사용됩니다.")
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)


class AIDecisionResult:
    """AI 의사결정 결과"""
    
    def __init__(self, action: str, confidence: float, reasoning: str = "", 
                 action_probs: Optional[Dict[str, float]] = None):
        self.action = action
        self.confidence = confidence
        self.reasoning = reasoning
        self.action_probs = action_probs or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (웹 전송용)"""
        return {
            'action': self.action,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'action_probs': self.action_probs,
            'timestamp': self.timestamp
        }


class AIModule:
    """
    AI 모듈 - 강화학습 기반 게임 AI
    
    Chloe가 구현할 주요 기능:
    1. PPO/DQN 정책 로드 및 추론
    2. 실시간 의사결정
    3. 자가 학습 데이터 수집
    4. 성능 모니터링
    """
    
    def __init__(self, model_path: Optional[str] = None, algorithm: str = "PPO"):
        """
        초기화
        
        Args:
            model_path: 훈련된 모델 경로
            algorithm: 사용할 알고리즘 ("PPO" 또는 "DQN")
        """
        self.model_path = model_path
        self.algorithm = algorithm
        # PyTorch가 없으면 device는 None (시뮬레이션 모드)
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        
        # 모델들
        self.policy_net = None
        self.value_net = None
        self.ppo_model = None
        self.dqn_model = None
        
        # 성능 추적
        self.decision_times = []
        self.action_history = []
        self.reward_history = []
        
        # RL 계측 (Chloe가 구현)
        self.rl_logger = None
        
        # 초기화
        self._initialize_model()
    
    def _initialize_model(self):
        """
        모델 초기화
        
        TODO for Chloe: 실제 PPO/DQN 모델 로드 구현
        """
        if self.model_path:
            # TODO: 실제 구현
            # if self.algorithm == "PPO":
            #     self.ppo_model = PPO.load(self.model_path)
            # elif self.algorithm == "DQN":
            #     self.dqn_model = DQN.load(self.model_path)
            
            print(f"🤖 [Chloe TODO] {self.algorithm} 모델 로드: {self.model_path}")
        else:
            # 기본 정책 네트워크 (시뮬레이션용) - PyTorch가 있을 때만
            if TORCH_AVAILABLE:
                self.policy_net = PolicyNetwork().to(self.device)
            print("⚠️ 모델 경로가 없습니다. 시뮬레이션 모드로 실행합니다.")
        
        # RL 계측 시스템 초기화
        # TODO: self.rl_logger = RLInstrumentationLogger("web_game_ai")
    
    def make_decision(self, game_state: Dict[str, Any]) -> AIDecisionResult:
        """
        게임 상태를 보고 행동 결정
        
        Args:
            game_state: 게임 엔진에서 받은 상태 정보
            
        Returns:
            AI 의사결정 결과
            
        TODO for Chloe: 실제 PPO/DQN 추론 구현
        """
        start_time = time.perf_counter()
        
        if self.ppo_model or self.dqn_model:
            # 실제 RL 모델 추론
            result = self._real_rl_decision(game_state)
        else:
            # 시뮬레이션 모드
            result = self._simulate_decision(game_state)
        
        # 성능 측정
        decision_time = time.perf_counter() - start_time
        self.decision_times.append(decision_time)
        self.action_history.append(result.action)
        
        return result
    
    def _simulate_decision(self, game_state: Dict[str, Any]) -> AIDecisionResult:
        """
        시뮬레이션된 AI 의사결정 (현재 구현)
        
        Chloe가 _real_rl_decision()으로 교체할 예정
        """
        # 간단한 휴리스틱 기반 의사결정
        player_y = game_state.get('player_y', 0.5)
        obstacle_y = game_state.get('obstacle_y', 0.0)
        obstacle_distance = game_state.get('obstacle_distance', 1.0)
        time_to_collision = game_state.get('time_to_collision', 10.0)
        
        # 의사결정 로직
        if time_to_collision < 1.0 and obstacle_distance < 0.3:
            if player_y > 0.7:  # 플레이어가 아래쪽에 있으면
                action = "jump"
                reasoning = "장애물이 가까워서 점프"
                confidence = 0.8
            else:
                action = "stay"
                reasoning = "이미 위쪽에 있어서 대기"
                confidence = 0.6
        else:
            # 랜덤 행동 (탐험)
            actions = ["stay", "jump", "left", "right"]
            weights = [0.4, 0.3, 0.15, 0.15]
            action = np.random.choice(actions, p=weights)
            reasoning = f"탐험적 행동: {action}"
            confidence = 0.5
        
        # 행동 확률 분포 (시뮬레이션)
        action_probs = {
            "stay": 0.4,
            "jump": 0.3,
            "left": 0.15,
            "right": 0.15
        }
        action_probs[action] += 0.2  # 선택된 행동의 확률 증가
        
        return AIDecisionResult(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            action_probs=action_probs
        )
    
    def _real_rl_decision(self, game_state: Dict[str, Any]) -> AIDecisionResult:
        """
        실제 강화학습 모델 의사결정
        
        TODO for Chloe: 이 함수를 구현하세요!
        
        구현 가이드:
        1. 게임 상태를 RL 모델 입력 형식으로 변환
        2. PPO 또는 DQN 추론 실행
        3. 행동 확률 분포 계산
        4. 최적 행동 선택
        5. 의사결정 근거 생성
        """
        try:
            # 상태 벡터 생성
            state_vector = self._create_state_vector(game_state)
            
            if self.algorithm == "PPO" and self.ppo_model:
                # TODO: PPO 추론
                # action, _states = self.ppo_model.predict(state_vector, deterministic=False)
                # action_probs = self._get_action_probabilities(state_vector)
                
                # 임시: 시뮬레이션 호출
                return self._simulate_decision(game_state)
                
            elif self.algorithm == "DQN" and self.dqn_model:
                # TODO: DQN 추론
                # action, _states = self.dqn_model.predict(state_vector, deterministic=False)
                # q_values = self._get_q_values(state_vector)
                
                # 임시: 시뮬레이션 호출
                return self._simulate_decision(game_state)
            
        except Exception as e:
            print(f"❌ RL 모델 추론 오류: {e}")
            # 오류 시 시뮬레이션으로 폴백
            return self._simulate_decision(game_state)
    
    def _create_state_vector(self, game_state: Dict[str, Any]) -> np.ndarray:
        """
        게임 상태를 RL 모델 입력 벡터로 변환
        
        TODO for Chloe: 상태 표현 최적화
        """
        # 8차원 상태 벡터 생성
        state_vector = np.array([
            game_state.get('player_x', 0.5),
            game_state.get('player_y', 0.5),
            game_state.get('player_vy', 0.0),
            game_state.get('on_ground', 0.0),
            game_state.get('obstacle_x', 0.0),
            game_state.get('obstacle_y', 0.0),
            game_state.get('obstacle_distance', 1.0),
            game_state.get('time_to_collision', 10.0)
        ], dtype=np.float32)
        
        return state_vector
    
    def update_reward(self, reward: float, done: bool = False):
        """
        보상 업데이트 (자가 학습용)
        
        TODO for Chloe: 온라인 학습 구현
        """
        self.reward_history.append(reward)
        
        if self.rl_logger:
            # TODO: RL 계측 시스템에 기록
            # self.rl_logger.log_step(reward, done)
            pass
        
        # 에피소드 종료 시 학습 (선택적)
        if done and len(self.reward_history) > 100:
            self._update_policy()
    
    def _update_policy(self):
        """
        정책 업데이트 (온라인 학습)
        
        TODO for Chloe: PPO/DQN 온라인 학습 구현
        """
        # TODO: 실제 정책 업데이트 구현
        # 1. 경험 버퍼에서 배치 샘플링
        # 2. 정책 그래디언트 계산
        # 3. 모델 파라미터 업데이트
        # 4. 성능 로깅
        
        print("🔄 [Chloe TODO] 정책 업데이트 실행")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        if not self.decision_times:
            return {}
        
        avg_decision_time = np.mean(self.decision_times)
        avg_reward = np.mean(self.reward_history) if self.reward_history else 0
        
        # 행동 분포 계산
        action_counts = {}
        for action in self.action_history:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'avg_decision_time_ms': avg_decision_time * 1000,
            'avg_reward': avg_reward,
            'total_decisions': len(self.action_history),
            'action_distribution': action_counts,
            'recent_actions': self.action_history[-10:],  # 최근 10개 행동
            'algorithm': self.algorithm
        }
    
    def reset_episode(self):
        """에피소드 초기화"""
        if self.rl_logger:
            # TODO: 에피소드 종료 로깅
            # self.rl_logger.log_episode_end(...)
            pass
        
        # 히스토리 초기화 (선택적)
        if len(self.action_history) > 1000:  # 메모리 관리
            self.action_history = self.action_history[-500:]
            self.reward_history = self.reward_history[-500:]
    
    def save_model(self, save_path: str):
        """
        모델 저장
        
        TODO for Chloe: 훈련된 모델 저장 구현
        """
        if self.ppo_model:
            self.ppo_model.save(save_path)
        elif self.dqn_model:
            self.dqn_model.save(save_path)
        else:
            # PyTorch 모델 저장
            if TORCH_AVAILABLE and self.policy_net:
                torch.save(self.policy_net.state_dict(), save_path)
        
        print(f"💾 모델 저장 완료: {save_path}")


# Chloe가 사용할 헬퍼 함수들
def create_reward_function(game_state: Dict[str, Any], action: str, next_state: Dict[str, Any]) -> float:
    """
    보상 함수 설계
    
    TODO for Chloe: 게임에 맞는 보상 함수 구현
    """
    reward = 0.0
    
    # 생존 보상
    if not next_state.get('game_over', False):
        reward += 1.0
    
    # 충돌 페널티
    if next_state.get('game_over', False):
        reward -= 100.0
    
    # 점수 증가 보상
    score_diff = next_state.get('score', 0) - game_state.get('score', 0)
    reward += score_diff * 10.0
    
    # 불필요한 행동 페널티 (선택적)
    if action in ["left", "right"] and game_state.get('obstacle_distance', 1.0) > 0.5:
        reward -= 0.1
    
    return reward


def analyze_failure_mode(game_state: Dict[str, Any], action: str) -> str:
    """
    실패 모드 분석
    
    Chloe가 디버깅용으로 사용할 수 있는 함수
    """
    if game_state.get('game_over', False):
        obstacle_distance = game_state.get('obstacle_distance', 1.0)
        time_to_collision = game_state.get('time_to_collision', 10.0)
        
        if obstacle_distance < 0.2 and action == "stay":
            return "회피 실패: 장애물이 가까운데 행동하지 않음"
        elif time_to_collision < 0.5 and action in ["left", "right"]:
            return "잘못된 회피: 점프 대신 좌우 이동"
        else:
            return "일반적인 충돌"
    
    return "정상"


# 사용 예시 (Chloe가 참고할 코드)
if __name__ == "__main__":
    # AI 모듈 초기화
    ai_module = AIModule(
        model_path="path/to/ppo_model.zip",  # Chloe가 훈련한 모델
        algorithm="PPO"
    )
    
    # 테스트 게임 상태
    test_state = {
        'player_x': 0.5,
        'player_y': 0.8,
        'player_vy': 0.0,
        'on_ground': 1.0,
        'obstacle_x': 0.6,
        'obstacle_y': 0.3,
        'obstacle_distance': 0.4,
        'time_to_collision': 2.0
    }
    
    # AI 의사결정
    decision = ai_module.make_decision(test_state)
    
    # 결과 출력
    print(f"선택된 행동: {decision.action}")
    print(f"신뢰도: {decision.confidence:.2f}")
    print(f"근거: {decision.reasoning}")
    
    # 성능 통계
    stats = ai_module.get_performance_stats()
    print(f"평균 의사결정 시간: {stats.get('avg_decision_time_ms', 0):.1f}ms")


# ============================================================================
# 난이도 레벨 시스템
# ============================================================================

class AIStrategy:
    """AI 전략 베이스 클래스"""
    
    def __init__(self, level: int, name: str):
        self.level = level
        self.name = name
    
    def make_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """의사결정 메서드 (서브클래스에서 구현)"""
        raise NotImplementedError


class Level1Strategy(AIStrategy):
    """
    Level 1 (Easy) - 간단한 휴리스틱
    
    전략:
    - 기본적인 메테오 회피만
    - 별은 무시
    - 중앙 유지 전략 약함
    """
    
    def __init__(self):
        super().__init__(level=1, name="Easy")
        self.DETECTION_RANGE = 200  # 메테오 감지 범위
        self.DANGER_RANGE = 100     # 위험 판정 범위
    
    def make_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """간단한 회피 로직"""
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
            
            # 플레이어보다 위쪽에 있고, X축 범위 내
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
            
            # 메테오 반대 방향으로 이동
            if meteor_center_x < player_center_x:
                return 'right'
            else:
                return 'left'
        
        return None  # 행동 없음


class Level2Strategy(AIStrategy):
    """
    Level 2 (Medium) - 고급 휴리스틱
    
    전략:
    - 메테오 회피 (향상된 로직)
    - 별 수집 전략
    - 중앙 유지
    - 용암 회피
    """
    
    def __init__(self):
        super().__init__(level=2, name="Medium")
        self.METEOR_DETECT_RANGE = 250
        self.METEOR_DANGER_RANGE = 150
        self.STAR_COLLECT_RANGE = 200
        self.EMERGENCY_RANGE = 80
    
    def make_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """향상된 의사결정 로직"""
        player = game_state.get('player', {})
        obstacles = game_state.get('obstacles', [])
        lava = game_state.get('lava', {})
        
        player_x = player.get('x', 480)
        player_y = player.get('y', 360)
        player_size = player.get('size', 50)
        player_center_x = player_x + player_size / 2
        
        WIDTH = 960
        HEIGHT = 720
        
        # 가장 가까운 메테오 & 별 찾기
        nearest_meteor = None
        nearest_meteor_dist = float('inf')
        nearest_star = None
        nearest_star_dist = float('inf')
        
        for obs in obstacles:
            obj_type = obs.get('type', 'meteor')
            obs_x = obs.get('x', 0)
            obs_y = obs.get('y', 0)
            obs_size = obs.get('size', 50)
            obs_center_x = obs_x + obs_size / 2
            
            # X축 중첩 체크
            x_overlap = abs(player_center_x - obs_center_x) < (player_size + obs_size) / 2 + 50
            
            if obj_type == 'meteor':
                if obs_y < player_y and x_overlap:
                    dist = abs(player_center_x - obs_center_x) + (player_y - obs_y) * 0.5
                    if dist < nearest_meteor_dist:
                        nearest_meteor_dist = dist
                        nearest_meteor = obs
            
            elif obj_type == 'star':
                if obs_y < player_y + 200:
                    dist = abs(player_center_x - obs_center_x) + abs(player_y - obs_y) * 0.3
                    if dist < nearest_star_dist:
                        nearest_star_dist = dist
                        nearest_star = obs
        
        # 우선순위 1: 긴급 메테오 회피
        if nearest_meteor and nearest_meteor_dist < self.EMERGENCY_RANGE:
            # 점프로 회피 시도
            if player_y >= HEIGHT - player_size - 10:
                return 'jump'
        
        # 우선순위 2: 메테오 회피
        if nearest_meteor and nearest_meteor_dist < self.METEOR_DANGER_RANGE:
            meteor_center_x = nearest_meteor['x'] + nearest_meteor.get('size', 50) / 2
            
            if meteor_center_x < player_center_x:
                if player_x + player_size < WIDTH - 20:
                    return 'right'
            else:
                if player_x > 20:
                    return 'left'
        
        # 우선순위 3: 별 수집
        if nearest_star and nearest_star_dist < self.STAR_COLLECT_RANGE:
            star_center_x = nearest_star['x'] + nearest_star.get('size', 30) / 2
            
            # 별 쪽으로 이동
            if star_center_x < player_center_x - 15:
                if player_x > 10:
                    return 'left'
            elif star_center_x > player_center_x + 15:
                if player_x + player_size < WIDTH - 10:
                    return 'right'
            
            # 별이 위쪽에 있으면 점프
            if nearest_star['y'] < player_y - 50 and player_y >= HEIGHT - player_size - 10:
                return 'jump'
        
        # 우선순위 4: 용암 회피
        if lava.get('state') in ['warning', 'active']:
            lava_zone_x = lava.get('zone_x', 0)
            lava_zone_width = lava.get('zone_width', 320)
            lava_zone_end = lava_zone_x + lava_zone_width
            
            # 플레이어가 용암 영역 안에 있으면
            if player_x + player_size > lava_zone_x and player_x < lava_zone_end:
                # 가장 가까운 안전 구역으로 이동
                if player_center_x < WIDTH / 2:
                    if player_x > 20:
                        return 'left'
                else:
                    if player_x + player_size < WIDTH - 20:
                        return 'right'
        
        # 우선순위 5: 중앙 유지
        center_x = WIDTH / 2
        if player_center_x < center_x - 100:
            if player_x + player_size < WIDTH - 20:
                return 'right'
        elif player_center_x > center_x + 100:
            if player_x > 20:
                return 'left'
        
        return None


class Level3Strategy(AIStrategy):
    """
    Level 3 (Hard) - PPO 모델 기반
    
    전략:
    - 학습된 PPO 모델 사용 (models/rl/ppo_agent.pt)
    - 모델이 없으면 최고급 휴리스틱으로 폴백
    """
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(level=3, name="Hard (PPO)")
        self.model_path = model_path
        self.ppo_model = None
        self.fallback_strategy = Level2Strategy()  # 폴백용 전략
        
        # PPO 모델 로드 시도
        self._load_ppo_model()
    
    def _load_ppo_model(self):
        """PPO 모델 로드"""
        if not self.model_path:
            print("⚠️ Level 3: PPO 모델 경로 없음, 휴리스틱으로 폴백")
            return
        
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                print(f"⚠️ Level 3: PPO 모델 파일 없음 ({self.model_path}), 휴리스틱으로 폴백")
                return
            
            # PyTorch 모델 로드 시도
            if TORCH_AVAILABLE:
                import torch
                self.ppo_model = torch.load(self.model_path, map_location='cpu')
                self.ppo_model.eval()
                print(f"✅ Level 3: PPO 모델 로드 성공 ({self.model_path})")
            else:
                print("⚠️ Level 3: PyTorch 없음, 휴리스틱으로 폴백")
        
        except Exception as e:
            print(f"⚠️ Level 3: PPO 모델 로드 실패 ({e}), 휴리스틱으로 폴백")
            self.ppo_model = None
    
    def make_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """PPO 모델 또는 폴백 전략"""
        # PPO 모델이 있으면 사용
        if self.ppo_model is not None:
            try:
                return self._ppo_decision(game_state)
            except Exception as e:
                print(f"⚠️ Level 3: PPO 추론 오류 ({e}), 휴리스틱으로 폴백")
        
        # 폴백: Level 2 전략 사용
        return self.fallback_strategy.make_decision(game_state)
    
    def _ppo_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """
        PPO 모델 기반 의사결정
        
        TODO for Chloe: 실제 PPO 추론 구현
        """
        # TODO: 게임 상태를 PPO 입력 형식으로 변환
        # state_vector = self._encode_state(game_state)
        
        # TODO: PPO 추론
        # with torch.no_grad():
        #     action_probs = self.ppo_model(state_vector)
        #     action_idx = torch.argmax(action_probs).item()
        
        # TODO: 행동 매핑
        # actions = [None, 'jump', 'left', 'right']
        # return actions[action_idx]
        
        # 임시: 폴백 사용
        return self.fallback_strategy.make_decision(game_state)


class Level4Strategy(AIStrategy):
    """
    Level 4 (Expert) - Ensemble 모델
    
    전략:
    - PPO + Vision 기반 앙상블
    - 여러 모델의 의사결정을 결합
    - 가장 높은 성능 목표
    """
    
    def __init__(self, ppo_model_path: Optional[str] = None, dqn_model_path: Optional[str] = None):
        super().__init__(level=4, name="Expert (Ensemble)")
        self.ppo_strategy = Level3Strategy(model_path=ppo_model_path)
        self.base_strategy = Level2Strategy()
        self.dqn_model_path = dqn_model_path
        self.dqn_model = None
        
        # DQN 모델 로드 시도 (선택적)
        self._load_dqn_model()
    
    def _load_dqn_model(self):
        """DQN 모델 로드 (선택적)"""
        if not self.dqn_model_path:
            return
        
        try:
            model_file = Path(self.dqn_model_path)
            if not model_file.exists():
                print(f"⚠️ Level 4: DQN 모델 파일 없음 ({self.dqn_model_path})")
                return
            
            if TORCH_AVAILABLE:
                import torch
                self.dqn_model = torch.load(self.dqn_model_path, map_location='cpu')
                self.dqn_model.eval()
                print(f"✅ Level 4: DQN 모델 로드 성공 ({self.dqn_model_path})")
        
        except Exception as e:
            print(f"⚠️ Level 4: DQN 모델 로드 실패 ({e})")
            self.dqn_model = None
    
    def make_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """앙상블 의사결정"""
        # 여러 전략의 결정을 수집
        decisions = []
        
        # PPO 전략
        ppo_action = self.ppo_strategy.make_decision(game_state)
        if ppo_action:
            decisions.append(('ppo', ppo_action, 0.5))  # 가중치 0.5
        
        # 휴리스틱 전략
        heuristic_action = self.base_strategy.make_decision(game_state)
        if heuristic_action:
            decisions.append(('heuristic', heuristic_action, 0.3))  # 가중치 0.3
        
        # DQN 전략 (있으면)
        if self.dqn_model is not None:
            # TODO: DQN 추론 구현
            # dqn_action = self._dqn_decision(game_state)
            # decisions.append(('dqn', dqn_action, 0.2))
            pass
        
        # 가중치 기반 투표
        if not decisions:
            return None
        
        # 간단한 앙상블: 가장 높은 가중치의 행동 선택
        # 실제 구현에서는 더 정교한 앙상블 방법 사용 가능
        decisions.sort(key=lambda x: x[2], reverse=True)
        return decisions[0][1]


class AILevelManager:
    """AI 난이도 레벨 관리자"""
    
    def __init__(self, ppo_model_path: Optional[str] = None, dqn_model_path: Optional[str] = None):
        """
        초기화
        
        Args:
            ppo_model_path: Level 3, 4에서 사용할 PPO 모델 경로
            dqn_model_path: Level 4에서 사용할 DQN 모델 경로 (선택적)
        """
        self.strategies = {
            1: Level1Strategy(),
            2: Level2Strategy(),
            3: Level3Strategy(model_path=ppo_model_path),
            4: Level4Strategy(ppo_model_path=ppo_model_path, dqn_model_path=dqn_model_path)
        }
        self.current_level = 1
    
    def set_level(self, level: int):
        """난이도 레벨 설정"""
        if level not in self.strategies:
            raise ValueError(f"Invalid level: {level}. Must be 1, 2, 3, or 4.")
        self.current_level = level
        print(f"🎮 AI 난이도: Level {level} ({self.strategies[level].name})")
    
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
