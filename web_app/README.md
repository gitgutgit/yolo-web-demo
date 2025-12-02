# 🌐 Distilled Vision Agent - Web Application

**브라우저에서 플레이 가능한 실시간 비전 기반 게임 AI**

GCP Cloud Run에 배포 가능한 Flask + SocketIO 웹 애플리케이션

## 🎮 기능

### **Human Mode** 🧑

- 브라우저에서 직접 게임 플레이
- 키보드 컨트롤 (SPACE: 점프, A/D: 이동)
- 실시간 점수 및 생존 시간 표시

### **AI Mode** 🤖

- AI 에이전트 자동 플레이 관찰
- **4단계 난이도 레벨 시스템** (Easy / Medium / Hard / Expert)
- 실시간 AI 결정 과정 표시
- 컴퓨터 비전 + 정책 네트워크 통합

### **실시간 모니터링** 📊

- FPS 및 성능 통계
- 리더보드 시스템
- WebSocket 기반 실시간 통신

## 🚀 로컬 실행

### 1. 의존성 설치

```bash
cd web_app
pip install -r requirements.txt
```

### 2. 개발 서버 실행

```bash
python app.py
```

### 3. 브라우저 접속

```
http://localhost:8080
```

## ☁️ GCP Cloud Run 배포

### 사전 준비

1. GCP 프로젝트 생성
2. Google Cloud SDK 설치
3. Docker 설치

### 자동 배포

```bash
# 프로젝트 ID를 입력하여 배포
./deploy.sh your-gcp-project-id

# 또는 수동으로 각 단계 실행
gcloud config set project your-gcp-project-id
gcloud services enable cloudbuild.googleapis.com run.googleapis.com
docker build -t gcr.io/your-gcp-project-id/distilled-vision-agent .
docker push gcr.io/your-gcp-project-id/distilled-vision-agent
gcloud run deploy distilled-vision-agent --image gcr.io/your-gcp-project-id/distilled-vision-agent --platform managed --allow-unauthenticated
```

### Cloud Build 자동 배포 (권장)

```bash
# GitHub 연동 후 자동 배포
gcloud builds submit --config cloudbuild.yaml
```

## 🏗️ 아키텍처

```
Frontend (HTML5 Canvas + JavaScript)
    ↕ WebSocket (SocketIO)
Flask Backend (Python)
    ├── Game Session Management
    ├── AI Decision Logic (Simulated)
    ├── Real-time State Updates
    └── Performance Monitoring
```

## 📁 프로젝트 구조

```
web_app/
├── app.py                 # Flask 메인 애플리케이션
├── templates/
│   └── index.html        # 게임 웹 페이지
├── static/
│   ├── css/style.css     # 스타일시트
│   └── js/game.js        # 게임 클라이언트 로직
├── requirements.txt      # Python 의존성
├── Dockerfile           # 컨테이너 이미지 빌드
├── cloudbuild.yaml      # GCP Cloud Build 설정
├── deploy.sh           # 배포 스크립트
└── README.md           # 이 파일
```

## 🎯 게임 컨트롤

### Human Mode

- **SPACE**: 점프/플랩
- **A** / **←**: 왼쪽 이동
- **D** / **→**: 오른쪽 이동

### 공통 컨트롤

- **H**: Human 모드 전환
- **I**: AI 모드 전환
- **R**: 게임 재시작

## 🔧 기술 스택

### Backend

- **Flask**: 웹 프레임워크
- **Flask-SocketIO**: 실시간 WebSocket 통신
- **Gunicorn + Eventlet**: 프로덕션 WSGI 서버

### Frontend

- **HTML5 Canvas**: 게임 렌더링
- **Socket.IO Client**: 실시간 통신
- **Vanilla JavaScript**: 게임 로직
- **CSS3**: 반응형 UI 디자인

### Infrastructure

- **GCP Cloud Run**: 서버리스 컨테이너 배포
- **GCP Container Registry**: 도커 이미지 저장
- **GCP Cloud Build**: CI/CD 파이프라인

## 📊 성능 최적화

- **실시간 FPS 모니터링**: 60 FPS 목표
- **WebSocket 최적화**: 최소 레이턴시 통신
- **Canvas 렌더링 최적화**: RequestAnimationFrame 사용
- **서버 리소스 관리**: 세션별 독립적 게임 상태

## 🤖 AI 난이도 레벨 시스템

<details>
<summary><strong>📊 4단계 AI Skill Level (클릭하여 펼치기)</strong></summary>

### Level 1: Easy 😊

**간단한 휴리스틱 전략**

- **구현 위치**: `web_app/modules/ai_module.py` - `Level1Strategy` 클래스
- **전략**: 기본적인 메테오 회피만
- **특징**:
  - 감지 범위: 200px
  - 위험 범위: 100px
  - 별(star) 수집 무시
  - 중앙 유지 전략 없음
- **사용 사례**: 초보 플레이어 시뮬레이션

### Level 2: Medium 😎

**고급 휴리스틱 전략**

- **구현 위치**: `web_app/modules/ai_module.py` - `Level2Strategy` 클래스
- **전략**: 메테오 회피 + 별 수집 + 용암 회피
- **특징**:
  - 메테오 감지 범위: 250px (향상)
  - 위험 범위: 150px
  - 별 수집 전략 추가
  - 용암 영역 회피
  - 중앙 유지 전략
- **사용 사례**: 숙련된 플레이어 시뮬레이션

### Level 3: Hard 🔥

**PPO 모델 기반 AI**

- **구현 위치**: `web_app/modules/ai_module.py` - `Level3Strategy` 클래스
- **전략**: 학습된 PPO 모델 사용
- **모델 경로**: `web_app/models/rl/ppo_agent.pt` (Chloe가 학습)
- **특징**:
  - PyTorch 기반 PPO 정책 네트워크
  - 모델이 없으면 Level 2 전략으로 자동 폴백
  - 게임 상태를 RL 입력 벡터로 변환
- **사용 사례**: 강화학습 AI 성능 평가

### Level 4: Expert ⭐

**Ensemble 모델**

- **구현 위치**: `web_app/modules/ai_module.py` - `Level4Strategy` 클래스
- **전략**: PPO + DQN + 휴리스틱 앙상블
- **모델 경로**:
  - PPO: `web_app/models/rl/ppo_agent.pt`
  - DQN: `web_app/models/rl/dqn_agent.pt` (선택적)
- **특징**:
  - 여러 모델의 의사결정을 가중치 기반으로 결합
  - PPO (가중치 0.5) + 휴리스틱 (0.3) + DQN (0.2)
  - 가장 높은 성능 목표
- **사용 사례**: 최고 성능 AI 벤치마크

### 📂 관련 파일 구조

```
web_app/
├── modules/
│   └── ai_module.py                    # AI 레벨 시스템 구현
│       ├── Level1Strategy              # Easy
│       ├── Level2Strategy              # Medium
│       ├── Level3Strategy              # Hard (PPO)
│       ├── Level4Strategy              # Expert (Ensemble)
│       └── AILevelManager              # 레벨 관리자
├── app.py                              # 백엔드 통합
│   └── ai_decision()                   # AI 의사결정 함수
└── templates/index.html                # 프론트엔드 난이도 선택 UI
```

### 🔧 레벨 변경 방법

1. **프론트엔드**: AI Mode 클릭 → 난이도 선택 모달에서 레벨 선택
2. **백엔드**: `game.ai_level` 변수로 관리 (1~4)
3. **의사결정**: `ai_level_manager.set_level()` → `make_decision()`

</details>

## 📊 데이터 수집 시스템

<details>
<summary><strong>💾 자동 데이터 수집 (클릭하여 펼치기)</strong></summary>

모든 게임 세션 (Human & AI)의 데이터가 자동으로 수집됩니다.

### 수집 위치

```
web_app/
├── collected_gameplay/                 # 훈련 데이터 (State-Action-Reward)
│   └── session_YYYYMMDD_HHMMSS_{mode}/
│       ├── metadata.json               # 세션 메타데이터
│       ├── states_actions.jsonl        # RL 데이터 (Chloe용)
│       └── bboxes.jsonl                # YOLO 라벨 (Jeewon용)
│
├── game_dataset/                       # YOLO 훈련 데이터셋
│   ├── images/train/                   # 게임 프레임 이미지
│   └── labels/train/                   # YOLO 포맷 라벨
│
└── data/                               # 클라우드 저장 데이터
    ├── gameplay/                       # 세션 데이터
    ├── frames/                         # 프레임 이미지
    └── leaderboard.json                # 리더보드
```

### 데이터 포맷

#### 1. `metadata.json` - 세션 메타데이터

```json
{
  "session_id": "abc123...",
  "mode": "human",
  "score": 150,
  "survival_time": 45.3,
  "total_frames": 1359,
  "timestamp": "2025-11-25T12:34:56",
  "player_name": "Larry"
}
```

#### 2. `states_actions.jsonl` - RL 훈련 데이터 (Chloe용)

```jsonl
{"frame": 0, "state": {...}, "action": "jump", "reward": 1.0, "done": false}
{"frame": 1, "state": {...}, "action": "stay", "reward": 1.0, "done": false}
```

#### 3. `bboxes.jsonl` - YOLO 라벨 데이터 (Jeewon용)

```jsonl
{"frame": 0, "objects": [{"class": "player", "x": 480, "y": 360, "w": 50, "h": 50}, ...]}
```

### Policy Distillation 절차

**Human 플레이 데이터 → AI 모델 학습**

1. **데이터 수집**:
   - 경로: `web_app/collected_gameplay/session_*_human/`
   - 자동 수집: `app.py` - `save_training_data()` 함수
2. **Chloe의 RL 학습**:

   ```python
   # states_actions.jsonl 로드
   import json

   states = []
   actions = []
   rewards = []

   with open('collected_gameplay/session_*/states_actions.jsonl') as f:
       for line in f:
           data = json.loads(line)
           states.append(data['state'])
           actions.append(data['action'])
           rewards.append(data['reward'])

   # PPO/DQN 학습
   # ...
   ```

3. **모델 저장**:
   - 저장 경로: `web_app/models/rl/ppo_agent.pt`
   - Level 3, 4에서 자동 로드

### 자동 Export

**YOLO 데이터셋 자동 생성**

- 구현: `web_app/yolo_exporter.py` - `YOLOExporter` 클래스
- 호출: `app.py` - `save_training_data()` 함수 내
- 출력: `web_app/game_dataset/` (YOLO 포맷)

</details>

## 🔮 향후 통합 계획

현재는 시뮬레이션된 AI이지만, 팀원들과 통합 시:

1. **Jeewon의 YOLOv8**: 실제 객체 탐지로 교체
2. **Chloe의 PPO/DQN**: 실제 강화학습 훈련 루프 통합
3. **실시간 학습**: 브라우저에서 AI 훈련 과정 관찰
4. **데이터 수집**: Human 플레이 데이터로 Policy Distillation (✅ 구현 완료)

## 🌐 배포 URL 예시

배포 완료 후 다음과 같은 URL에서 접속 가능:

```
https://distilled-vision-agent-xxxxx-uc.a.run.app
```

## 🎉 팀 정보

**Team Backward** - COMS W4995 Deep Learning for Computer Vision

- **Jeewon Kim (jk4864)**: YOLOv8 & System Architecture
- **Chloe Lee (cl4490)**: PPO/DQN & Reinforcement Learning
- **Minsuk Kim (mk4434)**: Web Development & Deployment

---

**🚀 브라우저에서 바로 플레이하고 AI와 경쟁해보세요!**
