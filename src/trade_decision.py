import torch
import pandas as pd
from src.ai_trainer import LSTMModel

def decide_trade(model_path, data, seq_length=20):
    """
    학습된 AI 모델을 기반으로 매매 신호를 결정
    - model_path: 학습된 모델 경로
    - data: 최신 데이터
    - seq_length: 시퀀스 길이 (기본값: 20)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 7
    hidden_size = 64
    output_size = 3
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    latest_data = data[['close', 'MA5', 'MA20', 'RSI', 'Upper_Band', 'Lower_Band', 'Band_Width']].dropna().values[-seq_length:]
    if len(latest_data) < seq_length:
        print("데이터가 부족하여 신호를 생성할 수 없습니다.")
        return "hold"
    
    input_data = torch.tensor(latest_data, dtype=torch.float32).unsqueeze(0).to(device)
    output = model(input_data)
    prediction = torch.argmax(output, dim=1).item()

    signals = {0: "sell", 1: "hold", 2: "buy"}
    signal = signals.get(prediction, "hold") 
    print(f"모델 예측 결과: {signal}")
    return signal

if __name__ == "__main__":
    decide_trade("models/model.pth", pd.read_csv("data/historical_data.csv"))