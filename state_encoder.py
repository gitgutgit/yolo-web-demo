# AI_model/state_encoder.py
import numpy as np

WIDTH = 960
HEIGHT = 720
PLAYER_SIZE = 50
OBSTACLE_SIZE = 50  # default

# ✅ [중요] 액션 순서: 환경 / PPO / 시각화 전부 이 순서로 통일
# 0: stay, 1: left, 2: right, 3: jump
ACTION_LIST = ["stay", "left", "right", "jump"]

STATE_DIM = 26  # Updated from 20 to 26 to include velocity (vx, vy) for 3 meteors

def encode_state(detections: list, game_state: dict = None) -> np.ndarray:
    """
    Encode YOLO detections into a state vector.
    
    Vector Layout (26 dims):
    [0-1]: Player (x, y)
    [2-6]: Meteor 1 (dx, dy, dist, vx, vy)  ← Added vx, vy
    [7-11]: Meteor 2 (dx, dy, dist, vx, vy)  ← Added vx, vy
    [12-16]: Meteor 3 (dx, dy, dist, vx, vy) ← Added vx, vy
    [17-19]: Star (dx, dy, dist)
    [20-22]: Lava (caution, exist, dx)
    [23]: Ground (1 if on ground)
    [24-25]: Reserved/Padding
    """
    # Initialize vector
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    
    # Find Player (Class 0)
    player = None
    for d in detections:
        if d['cls'] == 0:
            player = d
            break
            
    if player:
        vec[0] = player['x']  # Center X
        vec[1] = player['y']  # Center Y
        
        # On Ground check (approximate)
        if player['y'] + player['h'] / 2 >= 0.95:
            vec[23] = 1.0
    else:
        # Player not found (dead or missed)
        vec[0] = 0.5
        vec[1] = 0.5

    player_x = vec[0]
    player_y = vec[1]
    
    # --- Process Meteors (Class 1) ---
    meteors = []
    for d in detections:
        if d['cls'] == 1:
            dx = d['x'] - player_x
            dy = d['y'] - player_y
            dist = np.sqrt(dx**2 + dy**2)
            
            # Extract velocity from game_state if available
            vx, vy = 0.0, 0.0
            if game_state and 'obstacles' in game_state:
                # Find matching obstacle by position
                for obs in game_state['obstacles']:
                    if obs['type'] == 'meteor':
                        obs_x_norm = obs['x'] / WIDTH
                        obs_y_norm = obs['y'] / HEIGHT
                        if abs(obs_x_norm - d['x']) < 0.05 and abs(obs_y_norm - d['y']) < 0.05:
                            # Normalize velocity to [-1, 1] range
                            vx = np.clip(obs.get('vx', 0) / 10.0, -1, 1)
                            vy = np.clip(obs.get('vy', 5) / 10.0, -1, 1)
                            break
            
            meteors.append((dist, dx, dy, vx, vy))
            
    # Sort by distance (ascending) and take top 3
    meteors.sort(key=lambda x: x[0])
    
    # Fill Meteor Slots (Indices 2-16)
    for i in range(3):
        base_idx = 2 + i * 5  # Changed from 3 to 5 (dx, dy, dist, vx, vy)
        if i < len(meteors):
            dist, dx, dy, vx, vy = meteors[i]
            vec[base_idx] = np.clip(dx, -1, 1)
            vec[base_idx+1] = np.clip(dy, -1, 1)
            vec[base_idx+2] = np.clip(dist, 0, 1)
            vec[base_idx+3] = vx  # Velocity X
            vec[base_idx+4] = vy  # Velocity Y
        else:
            # No meteor found for this slot -> set dist to 1.0 (far)
            vec[base_idx+2] = 1.0

    # --- Process Star (Class 2) ---
    nearest_star_dist = 1.0
    for d in detections:
        if d['cls'] == 2:
            dx = d['x'] - player_x
            dy = d['y'] - player_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < nearest_star_dist:
                nearest_star_dist = dist
                vec[17] = np.clip(dx, -1, 1)
                vec[18] = np.clip(dy, -1, 1)
                vec[19] = np.clip(dist, 0, 1)
    if nearest_star_dist == 1.0:
        vec[19] = 1.0 # No star

    # --- Process Lava (Class 3, 4) ---
    for d in detections:
        if d['cls'] == 3:  # Caution Lava
            vec[20] = 1.0
            vec[22] = np.clip(d['x'] - player_x, -1, 1)
        elif d['cls'] == 4:  # Exist Lava
            vec[21] = 1.0
            vec[22] = np.clip(d['x'] - player_x, -1, 1)

    return vec
