# web_app/app.py

import os
import time
import base64
from datetime import datetime

import numpy as np
import torch
from ultralytics import YOLO

from flask import Flask, send_from_directory, jsonify
from flask_socketio import SocketIO, emit

from game_core import GameCore
from state_encoder import encode_state, ACTION_LIST, STATE_DIM
from ppo.agent import PPOAgent

# ==========================
# 기본 설정
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")          # fine-tuned YOLO
PPO_MODEL_PATH = os.path.join(BASE_DIR, "ppo_agent.pt")      # trained PPO

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""          # /index.html 로 접근 가능
)
socketio = SocketIO(app, cors_allowed_origins="*")

# 전역 객체들 (main에서 초기화)
yolo_model = None
ppo_agent = None

game = None
game_running = False
current_mode = "human"          # 'human' or 'ai'
current_ai_level = 2            # 1~4
last_action = "stay"
pending_jump = False
show_detections = True
current_sid = None              # Track which client is playing

start_time = 0.0
player_name = None

# 데이터 수집 카운터
collected_states_count = 0
collected_images_count = 0

# action 확률 (AI 모드일 때)
last_action_probs = None

# 리더보드 (메모리 버전, 필요하면 나중에 파일 저장으로 확장 가능)
leaderboard = []  # 각 항목: {player, score, time, mode, date}


# ==========================
# PPO 로더 (새/옛 포맷 둘 다 지원)
# ==========================

def load_ppo_for_web(model_path: str) -> PPOAgent:
    """watch_agent.py와 동일한 로직으로 PPO 체크포인트 로드."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO agent not found at {model_path}")
    print(f"✅ Loading PPO agent from {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # 옛날 포맷: lr 키가 있음 → agent.load 사용
    if "lr" in checkpoint:
        print("   📂 Old checkpoint format detected (has 'lr')")
        agent = PPOAgent.load(model_path)
        return agent

    # 새 포맷 (BC + PPO 튜닝 이후)
    print("   📂 New checkpoint format detected")
    state_dim = checkpoint.get("state_dim", STATE_DIM)
    action_dim = checkpoint.get("action_dim", len(ACTION_LIST))

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.0001,
        gamma=0.95,
        eps_clip=0.2,
        K_epochs=10,
    )

    if "policy_state_dict" in checkpoint:
        agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        agent.policy_old.load_state_dict(checkpoint["policy_state_dict"])
    if "value_net_state_dict" in checkpoint:
        agent.value_net.load_state_dict(checkpoint["value_net_state_dict"])

    print(f"   ✅ Loaded: state_dim={state_dim}, action_dim={action_dim}")
    return agent


# ==========================
# Flask 라우트 (HTML / 리더보드)
# ==========================

@app.route("/")
def index():
    """http://localhost:5000/ → index.html"""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/favicon.ico")
def favicon():
    fav = os.path.join(BASE_DIR, "favicon.ico")
    if os.path.exists(fav):
        return send_from_directory(BASE_DIR, "favicon.ico")
    return ("", 204)


@app.route("/api/leaderboard")
def api_leaderboard():
    """리더보드 JSON 반환 (시간/점수 순 정렬)."""
    # time 내림차순 → score 내림차순
    sorted_scores = sorted(
        leaderboard,
        key=lambda x: (-x.get("time", 0), -x.get("score", 0))
    )
    return jsonify({"scores": sorted_scores})


# ==========================
# YOLO 헬퍼
# ==========================

CLS2NAME = {
    0: "player",
    1: "meteor",
    2: "star",
    3: "caution_lava",
    4: "exist_lava",
}


def run_yolo_on_frame(frame_rgb):
    """
    GameCore.render() 로 얻은 RGB 프레임에 YOLO 적용.
    반환:
      - detections_for_state: encode_state용 (normalized)
      - detections_for_client: index.html에서 그릴용 (pixel bbox)
    """
    if yolo_model is None:
        return [], []

    # Ultralytics YOLO는 RGB numpy 바로 먹음
    results = yolo_model(frame_rgb, verbose=False)
    detections_for_state = []
    detections_for_client = []

    if len(results) == 0:
        return detections_for_state, detections_for_client

    r0 = results[0]
    H, W, _ = frame_rgb.shape

    boxes = r0.boxes
    for box in boxes:
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])

        # normalized xywh (0~1)
        x, y, w, h = box.xywhn[0].tolist()

        detections_for_state.append({
            "cls": cls_idx,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "conf": conf,
        })

        # pixel xyxy
        if hasattr(box, "xyxy"):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
        else:
            # xywhn 기준으로 변환
            cx = x * W
            cy = y * H
            pw = w * W
            ph = h * H
            x1 = cx - pw / 2
            y1 = cy - ph / 2
            x2 = cx + pw / 2
            y2 = cy + ph / 2

        class_name = CLS2NAME.get(cls_idx, "unknown")

        detections_for_client.append({
            "bbox": [x1, y1, x2, y2],
            "class_name": class_name,
            "conf": conf,
        })

    return detections_for_state, detections_for_client


# ==========================
# 상태 → 프론트엔드 payload 변환
# ==========================
from game_core import GameCore, WIDTH, HEIGHT, PLAYER_SIZE, OBSTACLE_SIZE, LAVA_CONFIG
def build_state_payload(state_dict, time_elapsed: float):
    """
    GameCore._get_state() 에서 나온 state_dict + 경과 시간(time_elapsed)을
    프론트(index.html)의 JS가 기대하는 형태로 변환해주는 함수.
    """
    global current_mode, collected_states_count, last_action_probs

    # 1) 플레이어
    player = state_dict.get("player", {})
    player_payload = {
        "x": float(player.get("x", 0)),
        "y": float(player.get("y", 0)),
        "vy": float(player.get("vy", 0)),
        # ⚠️ 이거 매우 중요: JS 쪽 render()에서 player.size를 쓰고 있음
        "size": float(player.get("size", PLAYER_SIZE)),
        "health": float(player.get("health", 100)),
    }

    # 2) 장애물 (메테오 / 별)
    obstacles_payload = []
    for o in state_dict.get("obstacles", []):
        obstacles_payload.append({
            "x": float(o.get("x", 0)),
            "y": float(o.get("y", 0)),
            "size": float(o.get("size", OBSTACLE_SIZE)),
            "type": o.get("type", "meteor"),
            "vx": float(o.get("vx", 0.0)),
            "vy": float(o.get("vy", 5.0)),
        })

    # 3) 용암 정보
    lava = state_dict.get("lava", {})
    lava_payload = {
        "state": lava.get("state", "inactive"),
        "zone_x": float(lava.get("zone_x", 0)),
        "zone_width": float(lava.get("zone_width", LAVA_CONFIG["zone_width"])),
        "height": float(lava.get("height", LAVA_CONFIG["height"])),
        # timer는 game_loop에서 넣어주거나 여기서 기본값 0.0
        "timer": float(lava.get("timer", 0.0)),
    }

    # 4) 기본 메타 정보
    frame = int(state_dict.get("frame", 0))
    score = int(state_dict.get("score", 0))

    payload = {
        "player": player_payload,
        "obstacles": obstacles_payload,
        "lava": lava_payload,
        "score": score,
        "time": float(time_elapsed),
        "frame": frame,
        "mode": current_mode,
        "collected_states_count": int(collected_states_count),
        "collected_images_count": 0,   # 지금은 안 쓰니까 0
    }

    # 5) PPO action probs (AI 모드에서만)
    if last_action_probs is not None:
        payload["action_probs"] = last_action_probs

    return payload

# ==========================
# 게임 루프 (백그라운드 태스크)
# ==========================

def game_loop():
    """
    30 FPS 정도로 계속 step() 하면서
    game_update, game_over 를 socket으로 보내는 루프.
    """
    global game_running, last_action, pending_jump
    global collected_states_count, last_action_probs
    global start_time, game, current_mode, player_name

    fps = 30.0
    dt = 1.0 / fps

    print("🎮 Game loop started")

    while game_running:
        if game is None:
            break

        # 1) 액션 결정
        action = "stay"
        action_probs = None
        det_client = []  # 클라이언트에 보낼 YOLO 박스

        if current_mode == "human":
            # jump는 한 프레임만
            if pending_jump:
                action = "jump"
                pending_jump = False
            else:
                action = last_action

        else:  # AI 모드
            # GameCore 렌더 → YOLO → state encoding → PPO
            frame_rgb = game.render()
            det_state, det_client = run_yolo_on_frame(frame_rgb)

            # encode_state 에 GameCore의 내부 상태 dict 전달
            game_state = game._get_state()
            state_vec = encode_state(det_state, game_state)

            # PPO 액션 선택 (eval)
            try:
                # action index
                action_idx = ppo_agent.select_action_eval(state_vec)
                action = ACTION_LIST[action_idx]

                # action probs (policy_old 통해 추출)
                with torch.no_grad():
                    s = torch.FloatTensor(state_vec).unsqueeze(0)
                    if next(ppo_agent.policy_old.parameters()).is_cuda:
                        s = s.cuda()
                    probs_tensor = ppo_agent.policy_old(s)
                    action_probs = probs_tensor.cpu().numpy()[0].tolist()
            except Exception as e:
                print(f"⚠️ PPO action selection error: {e}")
                action = "stay"
                action_probs = None

            # state_vec 하나 수집했다고 가정
            collected_states_count += 1

        # 2) 환경 step
        state_dict, reward, done, _ = game.step(action)

        # lava timer 넣어주기 (HTML에서 쓰도록)
        if "lava" in state_dict:
            # timer 제대로 계산하려면 상태에 따라 업데이트 해야 하지만
            # 일단 0.0 기본값 유지
            state_dict["lava"]["timer"] = 0.0

        # 3) 시간 계산
        time_elapsed = time.time() - start_time

        # 4) state payload build
        if current_mode == "ai":
            last_action_probs = action_probs
        else:
            last_action_probs = None

        payload = build_state_payload(state_dict, time_elapsed)

        # AI 모드일 때 YOLO 결과 client에 전달
        if current_mode == "ai":
            payload["detections"] = det_client

        # 5) 클라이언트로 전송
        if state_dict.get("frame", 0) % 30 == 0:
            print(f"[DEBUG] frame={state_dict.get('frame')} score={state_dict.get('score')}")

        # index.html 쪽에서 data.state || data 로 처리하니까
        # 여기서는 payload 그대로 보냄
        socketio.emit("game_update", payload, room=current_sid)

        # ❌ game_started 는 여기서 매 프레임 보내면 안 됨 → on_start_game 에서 한 번만 보냄
        # socketio.emit("game_started", payload)  # ← 이 줄은 삭제!

        # 6) 게임 오버 처리
        if done:
            game_running = False
            final_score = state_dict.get("score", 0)
            final_time = time_elapsed

            entry = {
                "player": (player_name or "AI") if current_mode == "ai" else (player_name or "Unknown"),
                "score": final_score,
                "time": final_time,
                "mode": current_mode,
                "date": datetime.now().isoformat(),
            }
            leaderboard.append(entry)

            # 상위 50개까지만 유지
            if len(leaderboard) > 50:
                leaderboard[:] = sorted(
                    leaderboard,
                    key=lambda x: (-x.get("time", 0), -x.get("score", 0))
                )[:50]

            # 상위 5개 내보내기
            top5 = sorted(
                leaderboard,
                key=lambda x: (-x.get("time", 0), -x.get("score", 0))
            )[:5]

            socketio.emit("game_over", {
                "score": final_score,
                "time": final_time,
                "player_name": player_name,
                "leaderboard": top5,
            }, room=current_sid)
            print(f"💀 Game over: score={final_score}, time={final_time:.1f}s, mode={current_mode}")
            break

        time.sleep(dt)

    print("🛑 Game loop ended")



# ==========================
# Socket.IO 이벤트 핸들러
# ==========================

@socketio.on("connect")
def on_connect():
    print("✅ Client connected")


@socketio.on("disconnect")
def on_disconnect():
    print("❌ Client disconnected")


@socketio.on("start_game")
def on_start_game(data):
    """
    data: {
      mode: 'human' | 'ai',
      player_name: str or null,
      ai_level: int (1~4)
    }
    """
    from flask import request
    
    global game, game_running, current_mode, current_ai_level
    global last_action, pending_jump, start_time, player_name
    global collected_states_count, collected_images_count, last_action_probs
    global current_sid

    mode = data.get("mode", "human")
    name = data.get("player_name")
    ai_level = int(data.get("ai_level", 2))
    
    # Track this client's session
    current_sid = request.sid

    print(f"🚀 start_game received: mode={mode}, player_name={name}, ai_level={ai_level}, sid={current_sid}")

    # 새 게임 초기화
    game = GameCore()
    state = game._get_state()

    game_running = True
    current_mode = mode
    current_ai_level = ai_level
    last_action = "stay"
    pending_jump = False
    player_name = name if mode == "human" else None
    collected_states_count = 0
    collected_images_count = 0
    last_action_probs = None
    start_time = time.time()

    # 초기 상태 전송 (to specific room)
    payload = build_state_payload(state, 0.0)
    socketio.emit("game_started", {"state": payload}, room=current_sid)

    # 백그라운드 게임 루프 시작
    socketio.start_background_task(game_loop)

    # ack 콜백 응답
    return {"status": "ok"}


@socketio.on("player_action")
def on_player_action(data):
    """
    Human mode에서 키 입력 이벤트.
    data: { action: 'left' | 'right' | 'jump' }
    """
    global last_action, pending_jump

    action = data.get("action", "stay")
    # print(f"🎮 player_action: {action}")

    if current_mode != "human":
        return

    if action == "jump":
        pending_jump = True
    elif action in ("left", "right", "stay"):
        last_action = action


@socketio.on("toggle_detections")
def on_toggle_detections():
    global show_detections
    show_detections = not show_detections
    print(f"👁️ YOLO detections {'ON' if show_detections else 'OFF'}")


@socketio.on("frame_capture")
def on_frame_capture(data):
    """
    index.html에서 10프레임마다 보내는 캔버스 이미지.
    data: { image: 'data:image/png;base64,...', frame: int }
    """
    global collected_images_count

    img_data = data.get("image")
    frame_idx = data.get("frame", 0)

    if not img_data:
        return

    # 'data:image/png;base64,' prefix 제거
    if img_data.startswith("data:image"):
        img_data = img_data.split(",")[1]

    try:
        img_bytes = base64.b64decode(img_data)
    except Exception as e:
        print(f"⚠️ Failed to decode frame image: {e}")
        return

    # 원하면 디스크에 저장해서 오프라인 학습용으로 쓸 수 있음
    # 여기서는 그냥 카운터만 증가
    collected_images_count += 1

    # 예: ./collected_frames/frame_000123.png 로 저장하고 싶다면:
    # save_dir = os.path.join(BASE_DIR, "collected_frames")
    # os.makedirs(save_dir, exist_ok=True)
    # filename = os.path.join(save_dir, f"frame_{frame_idx:06d}.png")
    # with open(filename, "wb") as f:
    #     f.write(img_bytes)


# ==========================
# 메인
# ==========================

if __name__ == "__main__":
    print("✅ Loading YOLO model:", YOLO_MODEL_PATH)
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print("✅ Loading PPO model:", PPO_MODEL_PATH)
    ppo_agent = load_ppo_for_web(PPO_MODEL_PATH)

    # Flask+SocketIO 서버 실행
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
