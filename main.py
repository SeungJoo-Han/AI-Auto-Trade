from src.data_collector import fetch_data
from src.ai_trainer import train_model, load_and_split_data
from src.backtester import backtest
from src.trade_decision import decide_trade
from src.trade_executor import execute_trade
import pandas as pd
import time

def main():
    #while(True):
    # 1. 데이터 수집 및 전처리
    ticker = "KRW-XRP"
    data_path = "data/historical_data.csv"
    model_save_path = "models/model.pth"
    
    #print("데이터를 수집 중입니다...")
    #data = fetch_data(ticker, interval="day", count=2800) 
    #data.to_csv(data_path)
    #print("데이터 수집 완료.")
    
    train_data, test_data = load_and_split_data(data_path)

    # 2. 모델 학습
    res = 0
    for i in range(10):
        seq = 10
        epochs = 10
        train_model(train_data, model_save_path, seq_length=seq ,epochs=epochs, hidden_size=128)

        # 3. 백테스팅
        print(f"=====Test: {i}=====")
        _, _, profit, _ = backtest(model_save_path, test_data, seq_length=seq)
        res += profit
    print(f"Avg Profit: {res/10}")

    # 4. 자동 매매
    #signal = decide_trade(model_save_path, pd.read_csv(data_path), seq_length=20)
    #execute_trade(ticker, signal)
    #time.sleep(60)

if __name__ == "__main__":
    main()