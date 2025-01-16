import torch
import pandas as pd
from src.ai_trainer import LSTMModel

import torch

def backtest(model_path, test_data, seq_length=20, initial_balance=1000000, trading_fee=0.05):
    """
    학습된 AI 모델을 기반으로 백테스팅 실행
    - model: 학습된 모델
    - test_data: 테스트 데이터
    - initial_balance: 초기 자본 (기본값: 1,000,000원)
    - trading_fee: 거래 수수료 (기본값: 0.05%)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 7
    hidden_size = 128
    output_size = 3
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    balance = initial_balance
    holdings = 0.0
    trade_log = []

    for i in range(len(test_data) - seq_length):
        input_data = torch.tensor(test_data[i:i+seq_length], dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_data)
        prediction = torch.argmax(output, dim=1).item()
        current_price = test_data[i + seq_length][0]

        if prediction == 2:  # Buy
            if balance > 0:
                buy_amount = balance * (1 - trading_fee) / current_price
                holdings += buy_amount
                balance = 0
                trade_log.append(( "Buy", current_price, holdings))
        elif prediction == 0:  # Sell
            if holdings > 0:
                sell_value = holdings * current_price * (1 - trading_fee)
                balance += sell_value
                trade_log.append(("Sell", current_price, holdings))
                holdings = 0

    final_balance = balance + holdings * test_data[-1][0]
    profit = final_balance - initial_balance
    profit_percentage = (profit / initial_balance) * 100

    print(f"최종 잔고: {final_balance:.2f}원")
    print(f"순이익: {profit:.2f}원")
    print(f"수익률: {profit_percentage:.2f}%")
    print("=== 거래 로그 ===")
    for log in trade_log:
        print(log)

    return final_balance, profit, profit_percentage, trade_log


if __name__ == "__main__":
    backtest("models/model.pth", pd.read_csv("data/historical_data.csv").values)