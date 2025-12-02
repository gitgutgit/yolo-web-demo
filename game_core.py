import random
import time
import numpy as np

# Game Constants
WIDTH = 960
HEIGHT = 720
PLAYER_SIZE = 50
OBSTACLE_SIZE = 50

# Object Types
OBJECT_TYPES = {
    'meteor': {
        'color': '#FF4444',
        'size': 50,
        'vy': 5,
        'reward': -100
    },
    'star': {
        'color': '#FFD700',
        'size': 30,
        'vy': 3,
        'score': 10,
        'reward': 20
    }
}

# Lava Config
LAVA_CONFIG = {
    'enabled': True,
    'warning_duration': 3.0,   # Caution 표시 시간 (이때는 닿아도 됨)
    'active_duration': 3.0,    # 실제 용암 지속 시간 (이때 닿으면 데미지)
    'interval': 10.0,          # 용암 발생 간격 (20초 → 10초로 줄임)
    'height': 120,
    'damage_per_frame': 3,
    'zone_width': 320
}

class GameCore:
    """
    Pure Python implementation of the game logic for RL training.
    Removes Flask, SocketIO, and other web-related dependencies.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game state."""
        self.player_x = WIDTH // 2
        self.player_y = HEIGHT // 2
        self.player_vy = 0
        self.obstacles = []
        self.score = 0
        self.game_over = False
        self.frame = 0
        self.start_time = time.time()
        
        # Lava state
        self.lava_state = 'inactive'  # inactive, warning, active
        self.lava_timer = LAVA_CONFIG['interval']
        self.lava_phase_timer = 0
        self.lava_zone_x = 0
        self.player_health = 100
        
        return self._get_state()

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (str): 'stay', 'left', 'right', 'jump'
            
        Returns:
            state (dict): Current game state
            reward (float): Reward for the action
            done (bool): Whether the game is over
            info (dict): Additional information
        """
        if self.game_over:
            return self._get_state(), 0, True, {}

        # 1. Apply Action
        self._apply_action(action)
        
        # 2. Update Physics (Gravity, Obstacles, Lava)
        self._update_physics()
        
        # 3. Update Score (1 point per second = 30 frames)
        self._update_score()
        
        # 4. Check Collisions & Calculate Reward
        reward, done = self._check_collisions()
        self.game_over = done
        
        # 5. Return Step Result
        return self._get_state(), reward, done, {}

    def _apply_action(self, action):
        """Apply player action."""
        speed = 10
        if action == 'left':
            self.player_x = max(0, self.player_x - speed)
        elif action == 'right':
            self.player_x = min(WIDTH - PLAYER_SIZE, self.player_x + speed)
        elif action == 'jump':
            # Jump only if on ground (simple logic)
            if self.player_y >= HEIGHT - PLAYER_SIZE - 5:
                self.player_vy = -20

    def _update_physics(self):
        """Update game physics."""
        # Gravity
        self.player_vy += 1
        self.player_y += self.player_vy
        
        # Ground Collision
        if self.player_y >= HEIGHT - PLAYER_SIZE:
            self.player_y = HEIGHT - PLAYER_SIZE
            self.player_vy = 0
            
        # Ceiling Collision
        if self.player_y < 0:
            self.player_y = 0
            self.player_vy = 0

        # Update Obstacles
        for obs in self.obstacles:
            obs['x'] += obs.get('vx', 0)
            obs['y'] += obs.get('vy', 5)
            
            # Wrap around screen
            if obs['x'] < -obs.get('size', OBSTACLE_SIZE):
                obs['x'] = WIDTH
            elif obs['x'] > WIDTH:
                obs['x'] = -obs.get('size', OBSTACLE_SIZE)

        # Remove off-screen obstacles
        self.obstacles = [o for o in self.obstacles if o['y'] < HEIGHT]
        
        # Spawn new obstacles
        if random.random() < 0.08:
            # Increase star chance: 10% -> 30%
            obj_type = 'star' if random.random() < 0.3 else 'meteor'
            obj_config = OBJECT_TYPES[obj_type]
            
            self.obstacles.append({
                'type': obj_type,
                'x': random.randint(0, WIDTH - obj_config['size']),
                'y': -obj_config['size'],
                'vx': random.randint(-3, 3),
                'vy': obj_config['vy'],
                'size': obj_config['size']
            })
            
        # Update Lava
        if LAVA_CONFIG['enabled']:
            self._update_lava()
            
        self.frame += 1

    def _update_score(self):
        """
        Update score based on survival time.
        1 point per second (30 frames = 1 point at 30 FPS)
        """
        if self.frame % 30 == 0:  # Every 30 frames (1 second)
            self.score += 1

    def _update_lava(self):
        """Update lava state machine."""
        dt = 1.0 / 30.0  # Assuming 30 FPS for simulation logic
        
        if self.lava_state == 'inactive':
            self.lava_timer -= dt
            if self.lava_timer <= 0:
                self.lava_state = 'warning'
                self.lava_phase_timer = LAVA_CONFIG['warning_duration']
                # Pick a random zone
                self.lava_zone_x = random.randint(0, WIDTH - LAVA_CONFIG['zone_width'])
                
        elif self.lava_state == 'warning':
            self.lava_phase_timer -= dt
            if self.lava_phase_timer <= 0:
                self.lava_state = 'active'
                self.lava_phase_timer = LAVA_CONFIG['active_duration']
                
        elif self.lava_state == 'active':
            self.lava_phase_timer -= dt
            if self.lava_phase_timer <= 0:
                self.lava_state = 'inactive'
                self.lava_timer = LAVA_CONFIG['interval']
                self.player_health = 100  # Reset health after lava phase

    def _check_collisions(self):
        """Check collisions and return reward/done."""
        reward = 0.1  # Base survival reward (increased from 0.02)
        done = False
        
        player_rect = [self.player_x, self.player_y, PLAYER_SIZE, PLAYER_SIZE]
        
        # --- [Tuning] Edge Penalty (Camping Penalty) ---
        if self.player_x < 100 or self.player_x > WIDTH - 150:
            reward -= 1.0
            
        # --- [Session 9] Center-Staying Reward (Anti-Camping) ---
        player_x_norm = self.player_x / WIDTH
        if 0.3 <= player_x_norm <= 0.7:
            reward += 0.5
            
        # Check Obstacles
        nearest_meteor_dist = float('inf')
        for obs in self.obstacles:
            if obs['type'] == 'meteor':
                dist = np.sqrt((obs['x'] - self.player_x)**2 + (obs['y'] - self.player_y)**2)
                if dist < nearest_meteor_dist:
                    nearest_meteor_dist = dist
        
        # --- [New] Dodge Reward ---
        if not hasattr(self, '_prev_meteor_dist'):
            self._prev_meteor_dist = nearest_meteor_dist
            
        if nearest_meteor_dist < 250:
            if nearest_meteor_dist > self._prev_meteor_dist + 1:
                reward += 10.0
        
        self._prev_meteor_dist = nearest_meteor_dist

        for obs in self.obstacles[:]:
            obs_rect = [obs['x'], obs['y'], obs['size'], obs['size']]
            if self._rect_overlap(player_rect, obs_rect):
                if obs['type'] == 'meteor':
                    reward = -100
                    done = True
                elif obs['type'] == 'star':
                    reward += 500
                    self.score += 50
                    self.obstacles.remove(obs)
                    
        # Check Lava
        if self.lava_state == 'active':
             lava_rect = [self.lava_zone_x, HEIGHT - LAVA_CONFIG['height'], LAVA_CONFIG['zone_width'], LAVA_CONFIG['height']]
             if self._rect_overlap(player_rect, lava_rect):
                 self.player_health -= LAVA_CONFIG['damage_per_frame']
                 reward -= 5
                 if self.player_health <= 0:
                     reward = -50
                     done = True
             else:
                 dist_to_lava = (HEIGHT - LAVA_CONFIG['height']) - (self.player_y + PLAYER_SIZE)
                 if 0 < dist_to_lava < 100:
                     reward += 1.0

        # Survival Reward
        reward += 0.25
                    
        return reward, done

    def _rect_overlap(self, r1, r2):
        """Check if two rectangles overlap (AABB collision)."""
        return not (r1[0] + r1[2] < r2[0] or r1[0] > r2[0] + r2[2] or
                    r1[1] + r1[3] < r2[1] or r1[1] > r2[1] + r2[3])

    def _get_state(self):
        """Return current game state dictionary."""
        return {
            'player': {
                'x': self.player_x,
                'y': self.player_y,
                'vy': self.player_vy,
                'health': self.player_health
            },
            'obstacles': [
                {
                    'x': o['x'], 
                    'y': o['y'], 
                    'size': o['size'], 
                    'type': o['type'],
                    'vx': o.get('vx', 0),
                    'vy': o.get('vy', 5)
                } 
                for o in self.obstacles
            ],
            'lava': {
                'state': self.lava_state,
                'zone_x': self.lava_zone_x,
                'height': LAVA_CONFIG['height'],
                'zone_width': LAVA_CONFIG['zone_width']
            },
            'score': self.score,
            'frame': self.frame
        }

    def render(self):
        """
        Render the current game state to an image (numpy array) matching index.html style EXACTLY.
        """
        import cv2
        import numpy as np
        
        # 1. Background Gradient (Top: #e3f2fd -> Bottom: #fff9c4)
        # #e3f2fd = RGB(227, 242, 253) -> BGR(253, 242, 227)
        # #fff9c4 = RGB(255, 249, 196) -> BGR(196, 249, 255)
        if not hasattr(self, '_bg_cache'):
            top_color = np.array([253, 242, 227], dtype=np.float32)
            bot_color = np.array([196, 249, 255], dtype=np.float32)
            
            # Create gradient column
            gradient_col = np.linspace(top_color, bot_color, HEIGHT).astype(np.uint8)
            # Tile it horizontally
            self._bg_cache = np.tile(gradient_col[:, np.newaxis, :], (1, WIDTH, 1))
            
        img = self._bg_cache.copy()
        
        # 2. Lava
        if self.lava_state != 'inactive':
            x1 = int(self.lava_zone_x)
            y1 = int(HEIGHT - LAVA_CONFIG['height'])
            x2 = int(x1 + LAVA_CONFIG['zone_width'])
            y2 = int(HEIGHT)
            
            if self.lava_state == 'warning':
                # Warning: Semi-transparent Red with flashing
                overlay = img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 69, 255), -1)  # BGR: OrangeRed
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                
                # Border (thick red)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                
                # Warning Text
                text = "WARNING LAVA"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(img, text, (x1 + (x2-x1)//2 - tw//2, y1 + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
            elif self.lava_state == 'active':
                # Active: Solid bright fill first (더 뚜렷하게!)
                # HTML처럼 진한 빨간색/주황색으로 채우기
                cv2.rectangle(img, (x1, y1), (x2, y2), (60, 76, 255), -1)  # BGR: #FF4C3C (밝은 빨강)
                
                # Gradient overlay for depth
                for row in range(y1, y2):
                    ratio = (row - y1) / (y2 - y1)
                    # #FF4500 -> #FF6347 -> #DC143C (더 진하게)
                    if ratio < 0.5:
                        r = int(255)
                        g = int(69 + ratio * 2 * 30)
                        b = int(0 + ratio * 2 * 71)
                    else:
                        r = int(220)
                        g = int(20)
                        b = int(60)
                    # 반투명 오버레이
                    overlay_row = img[row:row+1, x1:x2].copy()
                    cv2.line(overlay_row, (0, 0), (x2-x1, 0), (b, g, r), 1)
                    cv2.addWeighted(overlay_row, 0.5, img[row:row+1, x1:x2], 0.5, 0, img[row:row+1, x1:x2])
                
                # Wave effect (gold sinusoidal lines) - 더 두껍고 뚜렷하게
                wave_color = (0, 215, 255)  # Gold BGR
                wave_offset = (self.frame * 3) % 100
                pts = []
                for x in range(x1, x2, 5):
                    y_wave = y1 + int(np.sin((x + wave_offset) / 15) * 12)
                    pts.append([x, y_wave])
                if len(pts) > 1:
                    pts = np.array(pts, dtype=np.int32)
                    cv2.polylines(img, [pts], False, wave_color, 4)  # 더 두껍게 (3→4)
                    # 두 번째 파동 추가
                    pts2 = []
                    for x in range(x1, x2, 5):
                        y_wave = y1 + 20 + int(np.sin((x + wave_offset + 50) / 15) * 8)
                        pts2.append([x, y_wave])
                    pts2 = np.array(pts2, dtype=np.int32)
                    cv2.polylines(img, [pts2], False, (0, 200, 255), 3)
                
                # Bubbles (animated) - 더 많이
                for i in range(8):  # 5개 → 8개
                    bx = x1 + int((x2-x1) / 8 * i + ((self.frame / 2 + i * 15) % 25))
                    by = y1 + 30 + int(np.sin((self.frame / 80 + i)) * 25)
                    cv2.circle(img, (int(bx), int(by)), 10, (0, 140, 255), -1)  # 더 큰 버블
                    cv2.circle(img, (int(bx), int(by)), 6, (100, 200, 255), -1)  # 안쪽 밝은색
                
                # 🔥 FIRE TEXT - HTML처럼 텍스트 추가! (YOLO가 이걸로 구분할 수 있음)
                text = "FIRE LAVA"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                text_x = x1 + (x2-x1)//2 - tw//2
                text_y = y1 + 45
                # 텍스트 배경 (검은색 테두리)
                cv2.putText(img, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
                # 텍스트 (흰색)
                cv2.putText(img, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 테두리 (밝은 노란색/주황색)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 4)  # Orange border

        # 3. Player (with shadow and gradient)
        px, py = int(self.player_x), int(self.player_y)
        
        # Shadow (ellipse below player)
        shadow_cx = px + PLAYER_SIZE // 2
        shadow_cy = py + PLAYER_SIZE + 5
        cv2.ellipse(img, (shadow_cx, shadow_cy), (PLAYER_SIZE // 2, 8), 0, 0, 360, (100, 100, 100), -1)
        
        # Player Body (Vertical Gradient: #667eea -> #764ba2)
        # BGR: #667eea = (234, 126, 102), #764ba2 = (162, 75, 118)
        c1 = np.array([234, 126, 102], dtype=np.float32)  # Top color BGR
        c2 = np.array([162, 75, 118], dtype=np.float32)   # Bottom color BGR
        
        # Draw rounded rectangle with gradient
        for i in range(PLAYER_SIZE):
            ratio = i / PLAYER_SIZE
            color = tuple(((1-ratio)*c1 + ratio*c2).astype(int).tolist())
            # Round corners (simple approach: shrink width at top/bottom)
            corner_radius = 10
            if i < corner_radius:
                inset = corner_radius - int(np.sqrt(corner_radius**2 - (corner_radius - i)**2))
            elif i > PLAYER_SIZE - corner_radius:
                dist_from_bottom = PLAYER_SIZE - i
                inset = corner_radius - int(np.sqrt(corner_radius**2 - (corner_radius - dist_from_bottom)**2))
            else:
                inset = 0
            cv2.line(img, (px + inset, py + i), (px + PLAYER_SIZE - inset, py + i), color, 1)
            
        # Health Bar (above player)
        health_pct = max(0, self.player_health / 100.0)
        bar_x = px
        bar_y = py - 12
        bar_w = PLAYER_SIZE
        bar_h = 6
        
        # Background (dark)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        # Health bar color based on health
        if health_pct > 0.6:
            h_color = (80, 175, 76)   # Green #4CAF50 BGR
        elif health_pct > 0.3:
            h_color = (7, 193, 255)   # Yellow #FFC107 BGR
        else:
            h_color = (54, 67, 244)   # Red #F44336 BGR
            
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * health_pct), bar_y + bar_h), h_color, -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

        # 4. Obstacles (Stars and Meteors)
        for obs in self.obstacles:
            ox, oy = int(obs['x']), int(obs['y'])
            size = int(obs['size'])
            cx, cy = ox + size//2, oy + size//2
            radius = size // 2
            
            if obs['type'] == 'star':
                # ⭐ Star: 5-pointed star with very subtle shadow (matching HTML)
                # HTML uses ctx.shadowColor = 'rgba(255, 215, 0, 0.8)' with shadowBlur = 20
                # This creates a VERY subtle glow, almost invisible
                # We'll skip the glow circle entirely to match the clean look
                
                # Star shape - exactly matching HTML
                # HTML: angle = (Math.PI * i) / spikes, then cos(angle - Math.PI/2)
                points = []
                outer_r = size / 2
                inner_r = size / 4
                spikes = 5
                for i in range(spikes * 2):
                    r = outer_r if i % 2 == 0 else inner_r
                    angle = (np.pi * i) / spikes
                    x = cx + int(r * np.cos(angle - np.pi / 2))
                    y = cy + int(r * np.sin(angle - np.pi / 2))
                    points.append([x, y])
                points = np.array(points, dtype=np.int32)
                
                # Gradient fill: center #FFD700 (gold) to edge #FFA500 (orange)
                # Since OpenCV doesn't support gradient fill easily, 
                # we'll draw the star with gold and add a subtle orange tint
                cv2.fillPoly(img, [points], (0, 215, 255))  # Gold BGR #FFD700
                
                # Add subtle orange shading on bottom half for gradient effect
                # Create a smaller inner star with slightly orange tint
                inner_points = []
                for i in range(spikes * 2):
                    r = (outer_r * 0.6) if i % 2 == 0 else (inner_r * 0.6)
                    angle = (np.pi * i) / spikes
                    x = cx + int(r * np.cos(angle - np.pi / 2))
                    y = cy + int(r * np.sin(angle - np.pi / 2))
                    inner_points.append([x, y])
                inner_points = np.array(inner_points, dtype=np.int32)
                
                # Blend a brighter yellow in center for depth
                overlay = img.copy()
                cv2.fillPoly(overlay, [inner_points], (0, 230, 255))  # Brighter gold
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
                
            else:  # Meteor - EXACT match to index.html
                vx = obs.get('vx', 0)
                vy = obs.get('vy', 5)
                
                # Tail direction: opposite to velocity
                tail_len = radius * 3
                tail_angle = np.arctan2(-vy, -vx)
                tail_end_x = cx + int(np.cos(tail_angle) * tail_len)
                tail_end_y = cy + int(np.sin(tail_angle) * tail_len)
                
                # 🔥 Tail gradient (from transparent to bright)
                # Draw multiple overlapping triangles for gradient effect
                perp_angle = tail_angle + np.pi / 2
                for t in range(5):
                    t_ratio = t / 5
                    t_len = tail_len * (1 - t_ratio * 0.7)
                    t_end_x = cx + int(np.cos(tail_angle) * t_len)
                    t_end_y = cy + int(np.sin(tail_angle) * t_len)
                    t_width = radius * 0.6 * (1 - t_ratio * 0.5)
                    
                    p1 = (t_end_x, t_end_y)
                    p2 = (int(cx + np.cos(perp_angle) * t_width), int(cy + np.sin(perp_angle) * t_width))
                    p3 = (int(cx - np.cos(perp_angle) * t_width), int(cy - np.sin(perp_angle) * t_width))
                    triangle = np.array([p1, p2, p3], dtype=np.int32)
                    
                    # Color gradient: outer is transparent orange, inner is bright yellow-white
                    if t_ratio < 0.3:
                        color = (0, 69, 255)      # OrangeRed
                    elif t_ratio < 0.6:
                        color = (0, 140, 255)     # Orange
                    else:
                        color = (0, 215, 255)     # Gold
                    
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [triangle], color)
                    alpha = 0.3 + t_ratio * 0.5
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                
                # Fire particles along tail
                for i in range(5):
                    particle_dist = tail_len * 0.3 + np.random.random() * tail_len * 0.4
                    particle_angle = tail_angle + (np.random.random() - 0.5) * 0.5
                    particle_x = int(cx + np.cos(particle_angle) * particle_dist)
                    particle_y = int(cy + np.sin(particle_angle) * particle_dist)
                    particle_size = int(np.random.random() * 4 + 2)
                    particle_color = (0, int(150 + np.random.random() * 100), 255)  # Orange-yellow BGR
                    cv2.circle(img, (particle_x, particle_y), particle_size, particle_color, -1)
                
                # Outer flame glow (radial gradient around meteor)
                for glow_r in range(int(radius * 1.4), radius, -3):
                    alpha = 0.15
                    overlay = img.copy()
                    # Color: white center -> yellow -> orange -> red outer
                    ratio = (glow_r - radius) / (radius * 0.4)
                    if ratio < 0.3:
                        glow_color = (200, 255, 255)  # White-yellow
                    elif ratio < 0.6:
                        glow_color = (0, 215, 255)    # Gold
                    else:
                        glow_color = (0, 100, 255)    # Orange-red
                    cv2.circle(overlay, (cx, cy), glow_r, glow_color, -1)
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                
                # Meteor body (bumpy rock shape)
                # Create bumpy polygon instead of circle
                bumps = 10
                rock_points = []
                for i in range(bumps):
                    angle = (np.pi * 2 * i) / bumps
                    # Deterministic bump factor based on index (not random, for consistency)
                    bump_factor = 0.8 + np.sin(i * 3.7) * 0.15 + np.cos(i * 2.3) * 0.1
                    bump_radius = radius * bump_factor
                    x = int(cx + np.cos(angle) * bump_radius)
                    y = int(cy + np.sin(angle) * bump_radius)
                    rock_points.append([x, y])
                rock_points = np.array(rock_points, dtype=np.int32)
                
                # Rock gradient (dark brown to light brown)
                # #8B4513 (139, 69, 19) -> #654321 (101, 67, 33) -> #3E2723 (62, 39, 35) -> #1A1A1A (26, 26, 26)
                cv2.fillPoly(img, [rock_points], (19, 69, 139))  # Base brown BGR (#8B4513)
                
                # Add shading (darker on one side)
                shade_points = []
                for i in range(bumps // 2, bumps):
                    angle = (np.pi * 2 * i) / bumps
                    bump_factor = 0.8 + np.sin(i * 3.7) * 0.15 + np.cos(i * 2.3) * 0.1
                    bump_radius = radius * bump_factor * 0.9
                    x = int(cx + np.cos(angle) * bump_radius)
                    y = int(cy + np.sin(angle) * bump_radius)
                    shade_points.append([x, y])
                shade_points.append([cx, cy])
                if len(shade_points) > 2:
                    shade_points = np.array(shade_points, dtype=np.int32)
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [shade_points], (35, 39, 62))  # Dark brown BGR
                    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
                
                # Cracks (orange glowing lines) - exactly like HTML
                crack_color = (0, 100, 255)  # Orange BGR
                
                # Crack 1
                cv2.line(img, 
                        (int(cx - radius * 0.5), int(cy - radius * 0.4)),
                        (int(cx - radius * 0.1), int(cy - radius * 0.1)), 
                        crack_color, 2)
                cv2.line(img,
                        (int(cx - radius * 0.1), int(cy - radius * 0.1)),
                        (int(cx + radius * 0.2), int(cy + radius * 0.3)),
                        crack_color, 2)
                
                # Crack 2
                cv2.line(img,
                        (int(cx + radius * 0.4), int(cy - radius * 0.3)),
                        (int(cx + radius * 0.1), int(cy + radius * 0.1)),
                        crack_color, 2)
                cv2.line(img,
                        (int(cx + radius * 0.1), int(cy + radius * 0.1)),
                        (int(cx - radius * 0.3), int(cy + radius * 0.4)),
                        crack_color, 2)
                
                # Hot spot (bright white-yellow)
                hot_spot_x = int(cx - radius * 0.2)
                hot_spot_y = int(cy - radius * 0.1)
                hot_spot_r = int(radius * 0.15)
                cv2.circle(img, (hot_spot_x, hot_spot_y), hot_spot_r, (200, 255, 255), -1)  # White-yellow BGR
                
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
