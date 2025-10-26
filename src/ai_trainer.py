import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 시점의 예측값만 사용
        return out

def create_sequences(data, seq_length=20, future_window=5, threshold=0.01):
    """
    시퀀스 데이터와 레이블을 생성
    - data: 시계열 데이터
    - seq_length: 시퀀스 길이 (기본값: 20)
    - future_window: 미래 예측 대상 기간간 (기본값: 5)
    - threshold: 매수/매도 기준 (기본값: 1%)
    """
    x, y = [], []
    for i in range(len(data) - seq_length - future_window):
        x.append(data[i:i + seq_length]) # 시퀀스 데이터 생성
        
        current_close = data[i + seq_length - 1][0]
        future_prices = data[i + seq_length:i + seq_length + future_window, 0]
        future_avg_price = future_prices.mean()
        price_change = (future_avg_price - current_close) / current_close

        # 레이블 할당
        if price_change > threshold:  # 매수
            y.append(2)
        elif price_change < -threshold:  # 매도
            y.append(0)
        else:  # 홀드
            y.append(1)
    return np.array(x), np.array(y)

def load_and_split_data(data_path, split_ratio=0.8):
    """
    데이터를 로드하고 학습/테스트 데이터로 분리
    - data_path: CSV 파일 경로
    - split_ratio: 학습 데이터 비율 (기본값: 80%)
    """
    df = pd.read_csv(data_path)
    df = df.dropna()
    data = df[['close', 'MA5', 'MA20', 'RSI', 'Upper_Band', 'Lower_Band', 'Band_Width']].values

    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data


def train_model(train_data, model_save_path, seq_length=20, future_window=5, threshold=0.01, learning_rate=0.001, epochs=50, hidden_size=64):
    """
    LSTM 모델을 학습하고 저장
    - train_data: 학습 데이터
    - model_save_path: 모델 저장 경로
    - seq_length: 시퀀스 길이 (기본값: 20)
    - learning_rate: 학습률 (기본값: 0.001)
    - epochs: 에폭 수 (기본값: 50)
    - hidden_size: LSTM hidden size (기본값: 64)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train = create_sequences(train_data, seq_length, future_window, threshold)
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    model = LSTMModel(input_size=7, hidden_size=hidden_size, output_size=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        #print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), model_save_path)
    print(f"모델이 저장되었습니다: {model_save_path}")

    return model