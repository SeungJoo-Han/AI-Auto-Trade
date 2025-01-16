# AI 기반 암호화폐 자동 매매 시스템

**PyTorch**와 **PyUpbit**를 활용하여 암호화폐 매매 자동화를 구현한 시스템입니다. 데이터를 수집하고 AI 모델로 학습한 후, 실시간으로 거래 결정을 수행하며 백테스팅 기능도 제공합니다.

---

## 📂 프로젝트 구조

project/
│
├── data/
│   └── historical_data.csv       # 차트 데이터 저장
│
├── models/
│   └── model.pth                # 학습된 AI 모델
│
├── src/
│   ├── .env                      # API 키  
│   ├── data_collector.py         # 데이터 수집 및 전처리
│   ├── ai_trainer.py             # AI 모델 학습
│   ├── trade_decision.py         # 거래 결정 로직
│   ├── trade_executor.py         # 거래 요청 처리
│   └── backtester.py             # 백테스팅 모듈
│
└── main.py                       # 메인 실행 스크립트


---

## 🚀 주요 기능

1. **데이터 수집**:
   - PyUpbit를 통해 지정된 암호화폐의 데이터를 수집.
   - 이동평균(MA), RSI, 볼린저밴드 등의 지표 계산.

2. **AI 모델 학습**:
   - LSTM 기반의 딥러닝 모델 학습.
   - `close`, `MA5`, `MA20`, `RSI`와 같은 특성을 활용.

3. **백테스팅**:
   - 학습된 모델을 사용하여 과거 데이터를 기반으로 전략 평가.
   - 잔고, 순이익, 수익률 등의 결과 출력.

4. **실시간 거래**:
   - 실시간으로 데이터를 가져와 AI 예측 결과에 따라 매수/매도 수행.