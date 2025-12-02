# 질문 답변 - UI 여백 & YOLO 정확도

## 질문 1: AI 모드에서 여백이 생기는 이유

**현상:** Human 모드는 화면이 꽉 차는데, AI 모드에서는 여백이 생김

**원인:**
AI 모드일 때 "🧠 PPO Action Probabilities" 패널이 표시되면서 사이드 패널의 높이가 늘어납니다. 

`index.html`의 CSS를 보면:
```css
.game-wrapper {
    display: grid;
    grid-template-columns: 1fr 350px;  /* 게임:사이드패널 비율 */
    gap: 24px;
}
```

**해결 방법:**
1. **빠른 해결:** Action Probabilities 패널의 높이를 줄이거나 다른 패널과 통합
2. **CSS 조정:** `#gameCanvas`의 크기를 고정하고 `aspect-ratio: 4/3`을 유지
3. **레이아웃 변경:** 사이드 패널을 한 쪽에 고정하고 게임 화면은 항상 동일한 크기 유지

현재 CSS에서:
```css
#gameCanvas {
    width: 100%;    /* ← 부모 크기에 따라 변함 */
    height: auto;
    aspect-ratio: 4 / 3;
}
```

이 부분을 고정 크기로 바꾸면 해결됩니다.

---

## 질문 2: 실시간 YOLO 정확도가 떨어지는 이유

**현상:** 
- 정적 이미지 스크린샷: 메테오 90%+, 별 88%+
- 실시간 게임: 정확도 하락

**정상적인 현상입니다.** 이유는:

### 1. 모션 블러 (Motion Blur)
- 메테오와 별이 빠르게 움직이면서 프레임마다 약간씩 흐려짐
- YOLO는 선명한 이미지에서 훈련되었으므로 흐린 객체는 confidence가 낮아짐

### 2. 프레임 다양성
- 정적 이미지: 객체가 중앙, 전체가 보이는 "좋은" 각도
- 실시간: 객체가 화면 끝에 걸쳐있거나, 부분만 보이거나, 겹쳐있을 수 있음

### 3. 훈련 데이터 부족
- **200-300장은 YOLO fine-tuning에 적은 편입니다**
- 보통 클래스당 최소 500-1000장 권장
- 특히 다양한 위치, 크기, 각도의 데이터가 필요

### 4. 데이터셋 분포 차이
- Fine-tuning에 사용한 이미지들이 주로 어떤 상황인지에 따라 달라짐
- 예: 메테오가 화면 상단에만 있는 이미지로 훈련했다면, 하단에 있을 때 정확도 하락

### 개선 방법

#### 즉시 적용 가능:
1. **Confidence Threshold 낮추기** (현재 임계값 확인 필요)
   - `yolo_model(frame, conf=0.3)` 같이 낮은 threshold 사용
   
2. **NMS (Non-Maximum Suppression) 조정**
   - `yolo_model(frame, iou=0.4)` 

#### 장기적 개선:
1. **데이터 증강 (Data Augmentation)**
   - 더 많은 게임 프레임 수집 (500-1000장)
   - 다양한 위치, 크기의 객체 포함
   
2. **훈련 시 augmentation 강화**
   ```python
   # YOLO 훈련 시
   model.train(
       data='data.yaml',
       epochs=100,
       augment=True,  # 자동 augmentation
       mosaic=1.0,    # Mosaic augmentation
       ...
   )
   ```

3. **Hard Negative Mining**
   - 잘못 탐지하는 프레임 위주로 재훈련

### 결론
**현재 88-90%는 200-300장 훈련 기준으로 괜찮은 수치입니다.** 
게임이 플레이 가능하고 PPO가 학습할 수 있다면 충분합니다. 
더 높은 정확도가 필요하다면 데이터를 500-1000장으로 늘리는 것이 가장 효과적입니다.
