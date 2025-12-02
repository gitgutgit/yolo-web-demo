import os
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class YOLOExporter:
    """
    Exports game data to YOLOv8 format.
    Structure:
    game_dataset/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """
    
    def __init__(self, base_dir: str = "game_dataset"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images" / "train"
        self.labels_dir = self.base_dir / "labels" / "train"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data.yaml
        self._create_data_yaml()
        
    def _create_data_yaml(self):
        """Creates the data.yaml file required by YOLO."""
        data = {
            'path': str(self.base_dir.absolute()),
            'train': 'images/train',
            'val': 'images/train',  # Using train for val for now as we don't have a split yet
            'nc': 5,
            'names': ['player', 'meteor', 'star', 'caution_lava', 'exist_lava']
        }
        
        yaml_path = self.base_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
            
    def export_session(self, session_id: str, collected_states: List[Dict[str, Any]], frames_dir: Path):
        """
        Exports a game session to YOLO format.
        
        Args:
            session_id: The session ID.
            collected_states: List of game states collected during the game.
            frames_dir: Directory where raw frames are saved.
        """
        print(f"🚀 Starting YOLO export for session {session_id}...")
        
        # Use current date for naming
        current_date = datetime.now().strftime("%Y%m%d")
        
        # Use first 8 chars of session_id to ensure uniqueness per session
        short_sid = session_id[:8]
            
        exported_count = 0
        
        try:
            from PIL import Image
        except ImportError:
            print("⚠️ Pillow not found. Installing required...")
            Image = None

        for i, state_record in enumerate(collected_states):
            frame_num = state_record['frame']
            game_state = state_record['state']
            
            # 1. Locate the image file
            image_filename = f"frame_{frame_num:05d}.png"
            src_image_path = frames_dir / image_filename
            
            if not src_image_path.exists():
                found = list(frames_dir.rglob(image_filename))
                if found:
                    src_image_path = found[0]
                else:
                    continue
            
            # 2. Generate new filename: game_{date}_{session_id[:8]}_{frame}.jpg
            # Example: game_20251121_a1b2c3d4_00123.jpg
            dst_filename_base = f"game_{current_date}_{short_sid}_{frame_num:05d}"
            
            dst_image_name = f"{dst_filename_base}.jpg"
            dst_image_path = self.images_dir / dst_image_name
            
            # Convert and save as JPG
            if Image:
                try:
                    with Image.open(src_image_path) as img:
                        img = img.convert('RGB')
                        img.save(dst_image_path, 'JPEG', quality=95)
                except Exception as e:
                    print(f"❌ Image conversion failed for {src_image_path}: {e}")
                    continue
            else:
                # Fallback: just copy and rename extension
                shutil.copy2(src_image_path, dst_image_path)
            
            # 3. Create label file
            label_filename = f"{dst_filename_base}.txt"
            label_path = self.labels_dir / label_filename
            
            self._create_label_file(label_path, game_state)
            exported_count += 1
            
        print(f"✅ Exported {exported_count} frames to YOLO dataset.")

    def _create_label_file(self, file_path: Path, game_state: Dict[str, Any]):
        """Creates a YOLO format label file from game state."""
        # YOLO format: class_id x_center y_center width height (normalized 0-1)
        
        # Canvas dimensions
        WIDTH = 960
        HEIGHT = 720
        
        lines = []
        
        # Player (Class 0)
        player_x = game_state['player_x']
        player_y = game_state['player_y']
        player_size = 50 # From app.py PLAYER_SIZE
        
        # Normalize
        px = (player_x + player_size / 2) / WIDTH
        py = (player_y + player_size / 2) / HEIGHT
        pw = player_size / WIDTH
        ph = player_size / HEIGHT
        
        lines.append(f"0 {px:.6f} {py:.6f} {pw:.6f} {ph:.6f}")
        
        # Obstacles (Class 1: Meteor, Class 2: Star)
        for obs in game_state['obstacles']:
            obs_x = obs['x']
            obs_y = obs['y']
            obs_size = obs['size']
            obs_type = obs.get('type', 'meteor') # Default to meteor if type missing
            
            ox = (obs_x + obs_size / 2) / WIDTH
            oy = (obs_y + obs_size / 2) / HEIGHT
            ow = obs_size / WIDTH
            oh = obs_size / HEIGHT
            
            class_id = 1 # Default Meteor
            if obs_type == 'star':
                class_id = 2
            
            lines.append(f"{class_id} {ox:.6f} {oy:.6f} {ow:.6f} {oh:.6f}")
            
        # Lava (Class 3: caution_lava, Class 4: exist_lava)
        lava = game_state.get('lava')
        if lava:
            lava_state = lava['state']
            if lava_state in ['warning', 'active']:
                l_zone_x = lava['zone_x']
                l_height = lava['height']
                l_width = lava['zone_width']
                
                # Lava is at the bottom of the screen
                l_y = HEIGHT - l_height
                
                lx = (l_zone_x + l_width / 2) / WIDTH
                ly = (l_y + l_height / 2) / HEIGHT
                lw = l_width / WIDTH
                lh = l_height / HEIGHT
                
                class_id = 3 if lava_state == 'warning' else 4
                lines.append(f"{class_id} {lx:.6f} {ly:.6f} {lw:.6f} {lh:.6f}")
            
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))
