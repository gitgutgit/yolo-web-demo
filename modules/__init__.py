"""
Modules Package - Team Collaboration Modules

팀원별 모듈 패키지
각자 독립적으로 작업할 수 있도록 구조화

모듈 구조:
- game_engine: 공통 게임 로직 (수정 금지)
- cv_module: Jeewon Kim (jk4864) - 컴퓨터 비전
- ai_module: Chloe Lee (cl4490) - AI 정책
"""

from .game_engine import GameState, GameActions, GameObject, Player, Obstacle
from .cv_module import ComputerVisionModule, CVDetectionResult
from .ai_module import AIModule, AIDecisionResult

__all__ = [
    # Game Engine (공통)
    'GameState',
    'GameActions', 
    'GameObject',
    'Player',
    'Obstacle',
    
    # CV Module (Jeewon)
    'ComputerVisionModule',
    'CVDetectionResult',
    
    # AI Module (Chloe)
    'AIModule',
    'AIDecisionResult',
]

# 버전 정보
__version__ = "1.0.0"
__author__ = "Team Prof.Peter.backward()"
__contributors__ = [
    "Jeewon Kim (jk4864) - Computer Vision",
    "Chloe Lee (cl4490) - AI Policy", 
    "Minsuk Kim (mk4434) - Web Integration"
]
