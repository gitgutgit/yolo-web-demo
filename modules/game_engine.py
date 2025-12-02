"""
Game Engine Module - Core Game Logic

공통으로 사용하는 게임 로직
모든 팀원이 참조하지만 수정하지 않는 안전한 모듈

Author: Minsuk Kim (mk4434) - Base Implementation
Used by: All team members
"""

import time
import random
import uuid
from typing import Dict, List, Tuple, Optional, Any


class GameConfig:
    """게임 설정 상수들"""
    WIDTH = 640
    HEIGHT = 480
    FPS = 30
    PLAYER_SIZE = 40
    OBSTACLE_SIZE = 40
    PLAYER_SPEED = 8
    JUMP_STRENGTH = -15
    GRAVITY = 0.8
    OBSTACLE_SPEED = 7


class GameObject:
    """게임 오브젝트 기본 클래스"""
    
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.id = str(uuid.uuid4())
    
    def get_rect(self) -> Dict[str, float]:
        """충돌 검사용 사각형 반환"""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height
        }
    
    def collides_with(self, other: 'GameObject') -> bool:
        """다른 오브젝트와 충돌 검사"""
        rect1 = self.get_rect()
        rect2 = other.get_rect()
        
        return (rect1['x'] < rect2['x'] + rect2['width'] and
                rect1['x'] + rect1['width'] > rect2['x'] and
                rect1['y'] < rect2['y'] + rect2['height'] and
                rect1['y'] + rect1['height'] > rect2['y'])


class Player(GameObject):
    """플레이어 오브젝트"""
    
    def __init__(self):
        super().__init__(
            x=GameConfig.WIDTH // 2,
            y=GameConfig.HEIGHT - 80,
            width=GameConfig.PLAYER_SIZE,
            height=GameConfig.PLAYER_SIZE
        )
        self.velocity_y = 0
        self.on_ground = True
    
    def update_physics(self):
        """물리 엔진 업데이트"""
        # 중력 적용
        self.velocity_y += GameConfig.GRAVITY
        self.y += self.velocity_y
        
        # 바닥 충돌
        if self.y >= GameConfig.HEIGHT - GameConfig.PLAYER_SIZE:
            self.y = GameConfig.HEIGHT - GameConfig.PLAYER_SIZE
            self.velocity_y = 0
            self.on_ground = True
        else:
            self.on_ground = False
        
        # 좌우 경계 제한
        self.x = max(0, min(GameConfig.WIDTH - GameConfig.PLAYER_SIZE, self.x))
    
    def jump(self):
        """점프 액션"""
        if self.on_ground:
            self.velocity_y = GameConfig.JUMP_STRENGTH
            self.on_ground = False
    
    def move_left(self):
        """왼쪽 이동"""
        self.x -= GameConfig.PLAYER_SPEED
    
    def move_right(self):
        """오른쪽 이동"""
        self.x += GameConfig.PLAYER_SPEED
    
    def get_state_vector(self) -> Dict[str, float]:
        """AI가 사용할 상태 벡터"""
        return {
            'player_x': self.x / GameConfig.WIDTH,  # 정규화 (0-1)
            'player_y': self.y / GameConfig.HEIGHT,
            'player_vy': self.velocity_y / 20.0,  # 속도 정규화
            'on_ground': 1.0 if self.on_ground else 0.0
        }


class Obstacle(GameObject):
    """장애물 오브젝트"""
    
    def __init__(self, x: float):
        super().__init__(
            x=x,
            y=-GameConfig.OBSTACLE_SIZE,
            width=GameConfig.OBSTACLE_SIZE,
            height=GameConfig.OBSTACLE_SIZE
        )
    
    def update(self):
        """장애물 이동"""
        self.y += GameConfig.OBSTACLE_SPEED
    
    def is_off_screen(self) -> bool:
        """화면 밖으로 나갔는지 확인"""
        return self.y > GameConfig.HEIGHT
    
    def get_distance_to_player(self, player: Player) -> float:
        """플레이어와의 거리 계산"""
        dx = abs(self.x - player.x)
        dy = abs(self.y - player.y)
        return (dx ** 2 + dy ** 2) ** 0.5


class GameState:
    """게임 상태 관리"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """게임 상태 초기화"""
        self.player = Player()
        self.obstacles: List[Obstacle] = []
        self.score = 0
        self.game_over = False
        self.start_time = time.time()
        self.frame_count = 0
    
    def get_survival_time(self) -> float:
        """생존 시간 계산"""
        return time.time() - self.start_time
    
    def update(self):
        """게임 상태 업데이트"""
        if self.game_over:
            return
        
        # 플레이어 물리 업데이트
        self.player.update_physics()
        
        # 장애물 업데이트
        for obstacle in self.obstacles:
            obstacle.update()
        
        # 화면 밖 장애물 제거 및 점수 업데이트
        initial_count = len(self.obstacles)
        self.obstacles = [obs for obs in self.obstacles if not obs.is_off_screen()]
        self.score += initial_count - len(self.obstacles)
        
        # 새 장애물 생성 (2% 확률)
        if random.random() < 0.02:
            x_pos = random.randint(0, GameConfig.WIDTH - GameConfig.OBSTACLE_SIZE)
            self.obstacles.append(Obstacle(x_pos))
        
        # 충돌 검사
        for obstacle in self.obstacles:
            if self.player.collides_with(obstacle):
                self.game_over = True
                break
        
        self.frame_count += 1
    
    def handle_action(self, action: str):
        """플레이어 액션 처리"""
        if self.game_over:
            return
        
        if action == "jump":
            self.player.jump()
        elif action == "left":
            self.player.move_left()
        elif action == "right":
            self.player.move_right()
        # "stay"는 아무것도 하지 않음
    
    def get_state_for_ai(self) -> Dict[str, Any]:
        """AI가 사용할 전체 게임 상태"""
        player_state = self.player.get_state_vector()
        
        # 가장 가까운 장애물 정보
        if self.obstacles:
            nearest_obstacle = min(self.obstacles, 
                                 key=lambda obs: obs.get_distance_to_player(self.player))
            
            obstacle_state = {
                'obstacle_x': nearest_obstacle.x / GameConfig.WIDTH,
                'obstacle_y': nearest_obstacle.y / GameConfig.HEIGHT,
                'obstacle_distance': nearest_obstacle.get_distance_to_player(self.player) / 800.0,  # 정규화
                'time_to_collision': max(0, (nearest_obstacle.y - self.player.y) / GameConfig.OBSTACLE_SPEED / GameConfig.FPS)
            }
        else:
            obstacle_state = {
                'obstacle_x': 0.0,
                'obstacle_y': 0.0,
                'obstacle_distance': 1.0,  # 최대 거리
                'time_to_collision': 10.0  # 충분히 큰 값
            }
        
        return {**player_state, **obstacle_state}
    
    def get_state_for_web(self) -> Dict[str, Any]:
        """웹 클라이언트용 게임 상태"""
        return {
            'player': {
                'x': self.player.x,
                'y': self.player.y,
                'vy': self.player.velocity_y
            },
            'obstacles': [
                {'x': obs.x, 'y': obs.y, 'id': obs.id}
                for obs in self.obstacles
            ],
            'score': self.score,
            'survival_time': self.get_survival_time(),
            'game_over': self.game_over,
            'frame_count': self.frame_count
        }


# 게임 액션 상수
class GameActions:
    STAY = "stay"
    JUMP = "jump"
    LEFT = "left"
    RIGHT = "right"
    
    ALL_ACTIONS = [STAY, JUMP, LEFT, RIGHT]
    
    @classmethod
    def is_valid_action(cls, action: str) -> bool:
        """유효한 액션인지 확인"""
        return action in cls.ALL_ACTIONS
