# 🎉 Phase 2 완료: Cloud Storage 통합

## ✅ 완료된 작업

### 1️⃣ **리더보드 UI 개선**

- ✅ 초기 20개 표시 (기존 10개에서 증가)
- ✅ "Show More" 버튼 추가 (20개씩 추가 로드)
- ✅ 전체 개수 표시 ("Showing 20 of 38")
- ✅ `date` 필드 사용 (timestamp 오류 수정)

**파일**: `templates/index.html`

---

### 2️⃣ **게임 세션 Cloud Storage 저장**

#### 구현 내역

- ✅ `storage_manager.py`에 `save_gameplay_session()` 메서드 추가
- ✅ GCS와 로컬 fallback 모두 지원
- ✅ 날짜별 폴더 구조: `gameplay/sessions/2025-11-19/session_xxx.json`
- ✅ `app.py`에서 Storage Manager 사용

#### 저장 데이터

```json
{
  "session_id": "ABC123...",
  "mode": "human",
  "score": 98,
  "survival_time": 42.01,
  "total_frames": 1260,
  "final_state": {
    "player_x": 480,
    "player_y": 670,
    "obstacles_count": 5
  },
  "timestamp": "2025-11-19T15:47:43.231592",
  "player_name": "개쩐당"
}
```

**파일**: `storage_manager.py`, `app.py`

---

### 3️⃣ **이미지 프레임 저장 (CV 훈련용)**

#### 구현 내역

- ✅ `storage_manager.py`에 `save_frame_image()` 메서드 추가
- ✅ JavaScript에서 Canvas 캡처 (10프레임마다 = 3 FPS 샘플링)
- ✅ Base64 PNG로 인코딩 → Python 백엔드로 전송
- ✅ Cloud Storage 또는 로컬에 저장

#### 저장 경로

```
gs://distilled-vision-game-data/gameplay/frames/2025-11-19/ABC12345/
├── frame_00000.png
├── frame_00010.png
├── frame_00020.png
└── ...
```

**파일**: `storage_manager.py`, `app.py`, `templates/index.html`

---

## 📊 데이터 저장 구조

### GCP Cloud Storage 버킷 구조

```
gs://distilled-vision-game-data/
├── leaderboard/
│   └── leaderboard.json                  ← 리더보드 (실시간 업데이트)
│
├── gameplay/
│   ├── sessions/                         ← 게임 세션 메타데이터
│   │   └── 2025-11-19/
│   │       ├── session_20251119_153000_ABC12345.json
│   │       └── session_20251119_154500_DEF67890.json
│   │
│   └── frames/                           ← 게임 프레임 이미지 (CV 훈련용)
│       └── 2025-11-19/
│           ├── ABC12345/
│           │   ├── frame_00000.png
│           │   ├── frame_00010.png
│           │   └── ...
│           └── DEF67890/
│               ├── frame_00000.png
│               └── ...
```

---

## 🔢 데이터 용량 예상

### 1세션 (평균 30초 플레이)

- **메타데이터**: 1 KB
- **프레임 이미지**:
  - 30 FPS × 30초 = 900 프레임
  - 샘플링 (10프레임마다) = 90장
  - 90장 × 100KB = **9 MB**
- **합계**: **~9 MB/session**

### 100세션

- **총 용량**: ~900 MB
- **GCS 비용**: ~$0.02/월 (매우 저렴)

---

## 🚀 사용 방법

### 로컬 개발

```bash
cd web_app
source venv/bin/activate
python app.py
```

- ✅ 로컬 파일 시스템 사용 (`./data/`)
- ✅ 데이터 자동 저장 (게임 플레이 시)

### GCP 배포

```bash
cd web_app
./quick_deploy.sh
```

- ✅ 자동으로 Cloud Storage 사용
- ✅ 서버 재시작해도 데이터 유지
- ✅ 팀원들 어디서든 접근 가능

---

## 👥 팀원 데이터 접근

### Jay (YOLO 훈련용)

```bash
# 이미지 다운로드
gsutil -m cp -r gs://distilled-vision-game-data/gameplay/frames ./training_data/

# 특정 날짜만
gsutil -m cp -r gs://distilled-vision-game-data/gameplay/frames/2025-11-19 ./training_data/
```

### Chloe (RL 훈련용)

```bash
# 세션 메타데이터 다운로드
gsutil -m cp -r gs://distilled-vision-game-data/gameplay/sessions ./rl_training/
```

### Larry (데이터 증강 & 품질 관리)

```bash
# 전체 데이터 다운로드
gsutil -m cp -r gs://distilled-vision-game-data/gameplay ./data_processing/
```

---

## 🔧 설정 파일

### `.env` (로컬 개발)

```bash
ENVIRONMENT=development
LOCAL_DATA_DIR=./data
PORT=5002
```

### Cloud Run 환경 변수 (배포 시)

```bash
ENVIRONMENT=production
GCS_BUCKET_NAME=distilled-vision-game-data
```

---

## 📝 다음 단계 (선택사항)

### Phase 3: 고급 기능 (나중에)

- [ ] 프레임 캡처 On/Off 토글 버튼 (UI)
- [ ] State-Action-Reward 데이터도 Cloud Storage로
- [ ] Bbox 라벨링 자동 생성 (YOLO 포맷)
- [ ] 데이터셋 다운로드 API (`/api/dataset/download`)
- [ ] 학습된 모델 업로드 & 배포

---

## 🆘 문제 해결

### Q1. "google-cloud-storage not installed" 경고

→ **정상입니다!** 로컬에서는 자동으로 파일 시스템 사용

### Q2. GCP 배포 후 이미지 안 보임

```bash
# Cloud Run 서비스 계정 확인
gcloud run services describe distilled-vision-agent --region us-central1

# Storage Admin 권한 부여
gcloud projects add-iam-policy-binding vision-final-478501 \
  --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
  --role="roles/storage.admin"
```

### Q3. 로컬에서 GCS 테스트하고 싶음

```bash
# 서비스 계정 키 생성
gcloud iam service-accounts keys create credentials/gcp-key.json \
  --iam-account=game-storage-admin@vision-final-478501.iam.gserviceaccount.com

# .env 설정
echo "GOOGLE_APPLICATION_CREDENTIALS=./credentials/gcp-key.json" >> .env
echo "ENVIRONMENT=production" >> .env

# 재실행
python app.py
```

---

## 🎓 학습 목표 연계

이 프로젝트는 수업 내용과 완벽하게 연계됩니다:

### HW1-HW2: 이미지 분류

→ 게임 프레임으로 장애물/별/플레이어 분류 모델 훈련 가능

### HW3-HW4: CNN

→ YOLO로 실시간 객체 감지 (Jay)

### HW5: Transfer Learning

→ ResNet/ViT로 게임 상태 인식 모델

### Final Project: RL

→ 수집된 State-Action-Reward로 강화학습 (Chloe)

---

## 📊 현재 상태

| 기능                | 로컬 | GCP | 상태 |
| ------------------- | ---- | --- | ---- |
| 리더보드 저장       | ✅   | ✅  | 완료 |
| 게임 세션 저장      | ✅   | ✅  | 완료 |
| 이미지 프레임 저장  | ✅   | ✅  | 완료 |
| State-Action-Reward | ✅   | 📦  | 로컬 |
| Bbox 라벨링         | ✅   | 📦  | 로컬 |
| 모델 배포           | ⏳   | ⏳  | 대기 |

---

## 🎉 축하합니다!

**Phase 2 완료!** 이제 다음이 가능합니다:

1. ✅ 게임 플레이 → 자동으로 Cloud Storage에 저장
2. ✅ 서버 재시작해도 데이터 유지
3. ✅ 팀원들이 어디서든 훈련 데이터 접근
4. ✅ 이미지 데이터로 CV 모델 훈련 가능
5. ✅ 리더보드 더보기 기능

**다음 스텝**: GCP 배포 & 팀원들과 데이터 공유!
