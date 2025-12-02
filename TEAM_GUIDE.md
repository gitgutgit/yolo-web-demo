# 🎯 팀원별 작업 가이드

## 📁 **모듈화된 구조**

```
web_app/
├── app.py                 # 🔄 기존 버전 (참고용)
├── app_modular.py         # 🌟 새로운 모듈화 버전
├── modules/               # 📦 팀원별 모듈
│   ├── game_engine.py     # 🎮 공통 게임 로직 (수정 금지)
│   ├── cv_module.py       # 👁️ Jeewon 담당
│   ├── ai_module.py       # 🤖 Chloe 담당
│   └── web_session.py     # 🔗 Minsuk 담당
├── templates/
├── static/
└── requirements.txt
```

---

## 👁️ **Jeewon Kim (jk4864) - 컴퓨터 비전**

### **📁 담당 파일**

- `modules/cv_module.py`

### **🎯 목표**

- YOLOv8 기반 실시간 객체 탐지
- 60 FPS 달성 (≤16.7ms/frame)
- 웹 환경에서 안정적인 추론

### **📝 구현할 함수들**

#### 1. `_real_yolo_detection()` - 핵심 함수

```python
def _real_yolo_detection(self, frame: np.ndarray) -> List[CVDetectionResult]:
    """
    실제 YOLOv8 객체 탐지 구현

    TODO:
    1. 프레임 전처리 (640x640 리사이즈)
    2. YOLOv8 또는 ONNX 추론
    3. 후처리 (NMS, 신뢰도 필터링)
    4. CVDetectionResult 객체로 변환
    """
```

#### 2. `_initialize_model()` - 모델 로드

```python
def _initialize_model(self):
    """
    YOLOv8 모델 로드 및 ONNX 최적화

    TODO:
    1. self.model = YOLO(self.model_path)
    2. ONNX 변환 및 최적화
    3. 추론 세션 생성
    """
```

### **🔗 통합 포인트**

- `web_session.py`의 `_process_computer_vision()` 함수에서 호출됨
- 현재는 `_simulate_detection()`이 호출되고 있음
- Jeewon이 구현 완료하면 자동으로 실제 YOLOv8가 동작

### **🧪 테스트 방법**

```bash
# CV 모듈 단독 테스트
cd web_app/modules
python3 cv_module.py

# 전체 웹 앱 테스트
cd web_app
python3 app_modular.py
```

---

## 🤖 **Chloe Lee (cl4490) - AI 정책**

### **📁 담당 파일**

- `modules/ai_module.py`

### **🎯 목표**

- PPO/DQN 기반 실시간 의사결정
- 자가 학습 (Self-Play) 구현
- ≤5ms/decision 달성

### **📝 구현할 함수들**

#### 1. `_real_rl_decision()` - 핵심 함수

```python
def _real_rl_decision(self, game_state: Dict[str, Any]) -> AIDecisionResult:
    """
    실제 강화학습 모델 의사결정

    TODO:
    1. 게임 상태 → RL 입력 변환
    2. PPO/DQN 추론 실행
    3. 행동 확률 분포 계산
    4. 최적 행동 선택 및 근거 생성
    """
```

#### 2. `_initialize_model()` - 모델 로드

```python
def _initialize_model(self):
    """
    PPO/DQN 모델 로드

    TODO:
    1. self.ppo_model = PPO.load(self.model_path)
    2. 또는 self.dqn_model = DQN.load(self.model_path)
    3. RL 계측 시스템 초기화
    """
```

#### 3. `_update_policy()` - 온라인 학습

```python
def _update_policy(self):
    """
    정책 업데이트 (Self-Play)

    TODO:
    1. 경험 버퍼에서 배치 샘플링
    2. 정책 그래디언트 계산
    3. 모델 파라미터 업데이트
    """
```

### **🔗 통합 포인트**

- `web_session.py`의 `_get_ai_action()` 함수에서 호출됨
- 현재는 `_simulate_decision()`이 호출되고 있음
- Chloe가 구현 완료하면 자동으로 실제 PPO/DQN이 동작

### **🧪 테스트 방법**

```bash
# AI 모듈 단독 테스트
cd web_app/modules
python3 ai_module.py

# 전체 웹 앱 테스트 (AI 모드)
cd web_app
python3 app_modular.py
# 브라우저에서 "AI Mode" 선택
```

---

## 🔗 **Minsuk Kim (mk4434) - 웹 서버 & 통합**

### **📁 담당 파일**

- `app_modular.py` (메인 서버)
- `modules/web_session.py` (세션 관리)
- `modules/game_engine.py` (공통 로직)

### **✅ 완료된 작업**

- Flask-SocketIO 웹 서버
- 실시간 게임 세션 관리
- 팀원 모듈 통합 인터페이스
- GCP Cloud Run 배포

### **🔄 진행중인 작업**

- 성능 최적화 및 모니터링
- 팀원 모듈 통합 테스트
- 배포 자동화

---

## 🚀 **실행 및 테스트 가이드**

### **로컬 개발 환경**

```bash
# 1. 가상환경 설정
python3 -m venv venv
source venv/bin/activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 모듈화된 앱 실행
python3 app_modular.py

# 4. 브라우저 접속
open http://localhost:5000
```

### **팀원별 개발 워크플로우**

#### **Jeewon (CV 모듈)**

1. `modules/cv_module.py` 수정
2. YOLOv8 모델 훈련 및 저장
3. `_real_yolo_detection()` 구현
4. 단독 테스트: `python3 modules/cv_module.py`
5. 통합 테스트: `python3 app_modular.py`

#### **Chloe (AI 모듈)**

1. `modules/ai_module.py` 수정
2. PPO/DQN 모델 훈련 및 저장
3. `_real_rl_decision()` 구현
4. 단독 테스트: `python3 modules/ai_module.py`
5. 통합 테스트: `python3 app_modular.py` (AI 모드)

### **Git 브랜치 전략**

```bash
# 각자 브랜치 생성
git checkout -b jeewon-cv-module    # Jeewon
git checkout -b chloe-ai-module     # Chloe
git checkout -b minsuk-integration  # Minsuk

# 작업 완료 후 메인 브랜치에 병합
git checkout main
git merge jeewon-cv-module
git merge chloe-ai-module
```

---

## 📊 **성능 목표 및 측정**

### **전체 시스템**

- **웹 게임**: 30 FPS 안정적 동작
- **실시간 응답**: ≤100ms 지연시간

### **CV 모듈 (Jeewon)**

- **추론 속도**: ≤16.7ms/frame (60 FPS 가능)
- **탐지 정확도**: mAP ≥ 0.7
- **메모리 사용량**: ≤512MB

### **AI 모듈 (Chloe)**

- **의사결정 속도**: ≤5ms/decision
- **생존 시간**: 평균 120초 이상
- **학습 안정성**: 1000 에피소드 수렴

### **성능 모니터링**

```python
# 관리자 대시보드 접속
curl http://localhost:5000/admin

# 세션별 성능 확인
curl http://localhost:5000/health
```

---

## 🐛 **디버깅 및 문제 해결**

### **공통 문제**

1. **모듈 import 오류**: `sys.path` 확인
2. **의존성 오류**: `pip install -r requirements.txt`
3. **포트 충돌**: 다른 포트 사용 (`--port 5001`)

### **CV 모듈 (Jeewon)**

- **CUDA 오류**: CPU 모드로 폴백
- **메모리 부족**: 배치 크기 줄이기
- **ONNX 변환 실패**: PyTorch 버전 확인

### **AI 모듈 (Chloe)**

- **모델 로드 실패**: 경로 및 버전 확인
- **학습 불안정**: 하이퍼파라미터 조정
- **메모리 누수**: 경험 버퍼 크기 제한

---

## 📞 **소통 및 협업**

### **코드 리뷰**

- 각자 모듈 완성 후 팀원들과 리뷰
- 통합 테스트 전 상호 검토

### **이슈 트래킹**

- GitHub Issues 활용
- 라벨: `cv-module`, `ai-module`, `integration`

### **정기 미팅**

- 주 2회 진행 상황 공유
- 통합 이슈 해결 및 조율

---

## 🎯 **최종 목표**

**완성된 시스템:**

1. **실시간 웹 게임**: 브라우저에서 플레이 가능
2. **컴퓨터 비전**: YOLOv8 기반 60 FPS 객체 탐지
3. **AI 에이전트**: PPO/DQN 기반 자가 학습
4. **클라우드 배포**: GCP Cloud Run에서 안정적 서비스

**각자의 기여:**

- **Jeewon**: 실시간 비전 시스템
- **Chloe**: 지능적인 AI 에이전트
- **Minsuk**: 안정적인 웹 플랫폼

**함께 만드는 혁신적인 Vision-Based Game AI! 🚀**
