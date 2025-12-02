# YOLO Dataset Generation Explanation

This document explains how the game data is collected, processed, and exported into the YOLO format.

## Data Flow Overview

1.  **Game State Collection (`app.py`)**:
    - The `Game` class maintains the current state of the game.
    - In the `update()` method, the current state is captured **before** updating positions.
    - **Crucial Change**: 
        - We modified `app.py` to include the `type` of each obstacle ('meteor' or 'star').
        - We removed the limit on the number of obstacles (previously capped at 5).
        - We added `lava` state information to the collected data.
    - Code Location: `web_app/app.py` -> `Game.update()`

2.  **Data Saving (`app.py`)**:
    - When the game ends, `save_gameplay_session` calls `save_training_data`.
    - This function saves the raw session data and then triggers the `YOLOExporter`.

3.  **YOLO Export (`yolo_exporter.py`)**:
    - The `YOLOExporter` class reads the `collected_states`.
    - It iterates through each frame and its corresponding state.
    - **Image Handling**: It finds the saved frame image (PNG), converts it to JPG, and renames it to `game_{date}_{session_id}_{frame}.jpg`.
    - **Label Generation**: It calculates normalized coordinates for YOLO format.
    - **Class Distinction**:
        - **Class 0**: Player
        - **Class 1**: Meteor
        - **Class 2**: Star
        - **Class 3**: Caution Lava (Warning state)
        - **Class 4**: Exist Lava (Active state)
    - Code Location: `web_app/yolo_exporter.py` -> `_create_label_file()`

## File Structure & Naming

- **Images**: `game_dataset/images/train/game_{date}_{session_id}_{frame}.jpg`
- **Labels**: `game_dataset/labels/train/game_{date}_{session_id}_{frame}.txt`
- **Config**: `game_dataset/data.yaml`

## Classes

| Class ID | Name   | Description |
| :--- | :--- | :--- |
| 0 | player | The spaceship controlled by the user/AI |
| 1 | meteor | Dangerous obstacles to avoid |
| 2 | star   | Bonus items to collect |
| 3 | caution_lava | Warning state for lava (about to erupt) |
| 4 | exist_lava | Active lava zone (causes damage) |

## Coordinate Normalization

Coordinates are normalized to [0, 1] range as required by YOLO:
- `x_center = (x + width/2) / CANVAS_WIDTH`
- `y_center = (y + height/2) / CANVAS_HEIGHT`
- `width = object_width / CANVAS_WIDTH`
- `height = object_height / CANVAS_HEIGHT`

Canvas dimensions are assumed to be 960x720 (based on `app.py` logic).

## HOW to use?

1. Yolo train command in terminal 

yolo detect train model=yolov8n.pt data=./data.yaml epochs=50 imgsz=640

Recommend Ratio

val: 10 percent
train: 80 percent
test: 10 percent (unseen data)

or val:10 percent
train: 90 percent
no test 

2. Yolo detect command in terminal 

example: 

yolo detect predict model=best.pt source=images/val/game_20251122_BPrP-O6-_00790.jpg imgsz=640

