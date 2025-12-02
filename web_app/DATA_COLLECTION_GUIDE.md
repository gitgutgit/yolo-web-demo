# 📊 데이터 수집 및 훈련 가이드

## 🎯 개요

이 시스템은 GCP 웹 애플리케이션에서 실시간으로 게임 플레이 데이터를 수집하고, YOLO 및 RL 훈련을 위한 데이터셋으로 변환합니다.

---

## 🔄 데이터 흐름

```
사용자 플레이 (Human/AI Mode)
    ↓
프레임 + 상태 + 액션 수집
    ↓
collected_data/ 폴더에 저장
    ↓
Export API 호출
    ↓
YOLO/RL 훈련 데이터셋 생성
    ↓
Jeewon/Chloe가 모델 훈련
```

---

## 📁 폴더 구조

```
web_app/
├── collected_data/              # 수집된 원본 데이터 (Git에 푸시 안 됨)
│   └── session_{timestamp}/
│       ├── metadata.json        # 게임 세션 정보
│       ├── frames/             # 캡처된 프레임 이미지들
│       │   ├── frame_0000.png
│       │   ├── frame_0001.png
│       │   └── ...
│       └── states_actions.json  # 상태와 액션 로그
│
└── training_exports/            # 변환된 훈련 데이터 (Git에 푸시 안 됨)
    ├── yolo_dataset/           # YOLO 형식
    │   ├── images/
    │   ├── labels/
    │   └── dataset.yaml
    └── rl_dataset/             # RL 형식
        ├── observations.npy
        ├── actions.npy
        ├── rewards.npy
        └── metadata.json
```

> ⚠️ **주의**: `collected_data/`와 `training_exports/`는 `.gitignore`에 포함되어 있어 Git에 푸시되지 않습니다.

---

## 🎮 데이터 수집 방법

### 1️⃣ 웹 애플리케이션 실행

```bash
cd web_app
python app.py
```

브라우저에서 `http://localhost:5000` 접속

### 2️⃣ 게임 플레이

- **Human Mode**: 직접 플레이하여 전문가 데이터 수집
  - 키보드 조작: Space(점프), ←/→(이동)
  - 가능한 오래 생존
- **AI Mode**: AI의 결정을 관찰하며 데이터 수집
  - AI가 자동으로 플레이
  - 성공/실패 패턴 학습

### 3️⃣ 자동 저장

- 게임 종료 시 자동으로 `collected_data/`에 저장됨
- 프레임 이미지 + 상태/액션 로그가 세션별로 저장

---

## 📤 데이터 Export API

### 📊 수집 통계 확인

```bash
GET /api/data/stats
```

**응답 예시**:

```json
{
  "total_sessions": 42,
  "total_frames": 15420,
  "human_sessions": 25,
  "ai_sessions": 17,
  "avg_session_length": 367
}
```

### 🎯 YOLO 데이터셋 Export (Jeewon용)

```bash
POST /api/data/export/yolo
```

**생성되는 파일**:

- `training_exports/yolo_dataset/images/`: 프레임 이미지들
- `training_exports/yolo_dataset/labels/`: YOLO 형식 라벨 (.txt)
- `training_exports/yolo_dataset/dataset.yaml`: 데이터셋 설정 파일

**YOLO 라벨 형식** (각 줄):

```
<class_id> <x_center> <y_center> <width> <height>
```

- class_id: 0=player, 1=obstacle
- 좌표는 이미지 크기 대비 normalized (0~1)

**Jeewon이 사용하는 방법**:

```python
from ultralytics import YOLO

# 모델 훈련
model = YOLO('yolov8n.pt')
model.train(
    data='training_exports/yolo_dataset/dataset.yaml',
    epochs=100,
    imgsz=640
)
```

### 🤖 RL 데이터셋 Export (Chloe용)

```bash
POST /api/data/export/rl
```

**생성되는 파일**:

- `training_exports/rl_dataset/observations.npy`: 상태 벡터들 (numpy array)
- `training_exports/rl_dataset/actions.npy`: 액션들 (numpy array)
- `training_exports/rl_dataset/rewards.npy`: 보상들 (numpy array)
- `training_exports/rl_dataset/metadata.json`: 데이터셋 정보

**데이터 구조**:

- **observations**: shape (N, 8) - 각 타임스텝의 상태 벡터
  - [player_x, player_y, velocity_y, next_obstacle_x, next_obstacle_y, obstacle_width, obstacle_height, gap_size]
- **actions**: shape (N,) - 액션 인덱스 (0=nothing, 1=jump, 2=left, 3=right)
- **rewards**: shape (N,) - 각 타임스텝의 보상

**Chloe가 사용하는 방법**:

```python
import numpy as np
from stable_baselines3 import PPO

# 데이터 로드
obs = np.load('training_exports/rl_dataset/observations.npy')
actions = np.load('training_exports/rl_dataset/actions.npy')
rewards = np.load('training_exports/rl_dataset/rewards.npy')

# Policy Distillation (Imitation Learning)
# 또는 PPO/DQN으로 Self-Play
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
```

---

## 🔧 서버 코드 통합

### `app.py`의 핵심 부분

```python
from training_data_collector import TrainingDataCollector

# 초기화
data_collector = TrainingDataCollector()

# 게임 종료 시 자동 저장
@socketio.on('save_gameplay_data')
def handle_save_gameplay_data(data):
    session_id = data_collector.save_gameplay_session(
        frames=data['frames'],
        states=data['states'],
        actions=data['actions'],
        mode=data['mode'],
        final_score=data['score']
    )
    emit('data_saved', {'session_id': session_id})

# Export API
@app.route('/api/data/export/yolo', methods=['POST'])
def export_yolo():
    dataset_path = data_collector.export_for_yolo()
    return jsonify({'status': 'success', 'path': dataset_path})

@app.route('/api/data/export/rl', methods=['POST'])
def export_rl():
    dataset_path = data_collector.export_for_rl()
    return jsonify({'status': 'success', 'path': dataset_path})
```

---

## 🎯 팀원별 워크플로우

### 👨‍💻 Jeewon (CV Part)

1. **웹에서 데이터 수집**: Human/AI Mode로 충분한 게임 플레이
2. **YOLO 데이터셋 생성**: `POST /api/data/export/yolo`
3. **모델 훈련**:
   ```bash
   cd training_exports/yolo_dataset
   yolo train data=dataset.yaml model=yolov8n.pt epochs=100
   ```
4. **ONNX로 변환**: 실시간 추론을 위해 최적화
   ```python
   from src.deployment.onnx_optimizer import ONNXModelOptimizer
   optimizer = ONNXModelOptimizer()
   optimizer.convert_yolo_to_onnx('best.pt', 'yolo_optimized.onnx')
   ```
5. **`cv_module.py`에 통합**: 실제 YOLO 검출 구현

### 👩‍💻 Chloe (RL Part)

1. **웹에서 데이터 수집**: Human Mode로 전문가 플레이 수집
2. **RL 데이터셋 생성**: `POST /api/data/export/rl`
3. **Policy Distillation (Imitation)**:
   ```python
   # observations를 이용해 전문가를 모방하도록 학습
   from stable_baselines3.common.policies import ActorCriticPolicy
   # Supervised Learning으로 초기 정책 훈련
   ```
4. **Self-Play PPO**:
   ```python
   from stable_baselines3 import PPO
   model = PPO('MlpPolicy', env, verbose=1)
   model.learn(total_timesteps=1000000)
   model.save('ppo_agent')
   ```
5. **`ai_module.py`에 통합**: 실제 RL 정책 구현

### 🛠️ Larry (Deployment & Optimization)

1. **데이터 수집 모니터링**: `GET /api/data/stats`로 상태 확인
2. **Augmentation 적용**: Jeewon의 데이터를 더 robust하게 만들기
   ```python
   from src.data.augmentation import GameFrameAugmenter
   augmenter = GameFrameAugmenter()
   # YOLO 훈련 전에 데이터 증강
   ```
3. **성능 프로파일링**: 실시간 추론 속도 측정 (≤16.7ms 목표)
4. **ONNX 최적화**: 모델들을 60 FPS로 실행 가능하게 최적화
5. **문서화 및 Git 관리**: 팀 협업 지원

---

## 📈 데이터 품질 체크리스트

### ✅ 좋은 데이터셋을 위한 조건

- [ ] **다양한 시나리오**: 다양한 장애물 패턴과 속도
- [ ] **충분한 샘플 수**:
  - YOLO: 최소 500 프레임 (더 많을수록 좋음)
  - RL: 최소 50,000 타임스텝 (전문가 데이터)
- [ ] **균형 잡힌 액션 분포**: jump, left, right, nothing이 모두 포함
- [ ] **성공/실패 모두 포함**: 오래 생존한 게임 + 빨리 죽은 게임
- [ ] **Human 데이터 우선**: AI보다 사람의 플레이가 더 좋은 전문가 데이터

---

## 🚨 주의사항

### 🔒 보안

- **GCP 서비스 계정 키 (`.json`)는 절대 Git에 푸시하지 않기**
  - `.gitignore`에 이미 포함됨
  - 로컬에만 보관

### 💾 용량 관리

- `collected_data/`는 빠르게 커질 수 있음
- 필요 없는 세션은 주기적으로 삭제:
  ```bash
  rm -rf collected_data/session_20240101_*
  ```

### 🔄 데이터 동기화

- Export된 데이터셋은 팀원들끼리 공유 필요:
  - Google Drive 또는
  - GCS (Google Cloud Storage) 버킷

---

## 🎉 전체 워크플로우 예시

```bash
# 1. 웹 앱 실행
cd web_app
python app.py

# 2. 브라우저에서 게임 플레이 (Human Mode)
# → 자동으로 collected_data/에 저장됨

# 3. 통계 확인
curl http://localhost:5000/api/data/stats

# 4. YOLO 데이터셋 생성 (Jeewon)
curl -X POST http://localhost:5000/api/data/export/yolo

# 5. RL 데이터셋 생성 (Chloe)
curl -X POST http://localhost:5000/api/data/export/rl

# 6. 각자 모델 훈련
# Jeewon: YOLOv8 훈련 → ONNX 변환
# Chloe: PPO/DQN 훈련
# Larry: 성능 최적화 및 통합

# 7. 웹 앱에 통합
# → cv_module.py, ai_module.py에 훈련된 모델 적용
```

---

## 📚 관련 문서

- [TEAM_GUIDE.md](TEAM_GUIDE.md): 팀 통합 가이드
- [README.md](../README.md): 프로젝트 전체 개요
- [TEAM_CHECKLIST.md](../TEAM_CHECKLIST.md): 팀원별 체크리스트

---

## ❓ FAQ

**Q: 데이터 수집은 언제 자동으로 저장되나요?**

- A: 게임 종료 시 자동으로 `collected_data/`에 저장됩니다.

**Q: YOLO 라벨이 없는데 어떻게 생성되나요?**

- A: 현재 게임 상태(player, obstacle 위치)를 기반으로 자동 생성됩니다.

**Q: RL 데이터의 reward는 어떻게 계산되나요?**

- A: 생존 시간 + 장애물 통과 + 충돌 패널티로 계산됩니다.

**Q: GCP에 배포된 앱에서도 데이터가 수집되나요?**

- A: 네! Cloud Run에서도 동일하게 작동합니다. 단, 저장 용량 제한에 주의하세요.

---

**마지막 업데이트**: 2025-11-18  
**작성자**: Team Prof.Peter.backward()
