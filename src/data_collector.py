import pyupbit
import pandas as pd

def fetch_data(ticker, interval="day", count=1000):
    """
    지정된 티커의 데이터를 가져와서 MA5, MA20, RSI, 볼린저 밴드를 추가
    - ticker: 암호화폐 티커 (예: "KRW-XRP")
    - interval: 데이터 간격 (예: "minute1", "day")
    - count: 데이터 개수
    """
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)

    df['MA5'] = df['close'].rolling(window=5).mean()  # 이동평균선 5
    df['MA20'] = df['close'].rolling(window=20).mean()  # 이동평균선 20
    df['RSI'] = calculate_rsi(df['close'])
    _, df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df['close']) # 볼린저 밴드 계산
    df['Band_Width'] = df['Upper_Band'] - df['Lower_Band']  # 밴드 폭 추가

    return df

def calculate_rsi(series, period=14):
    """
    RSI 계산
    - series: 종가 데이터
    - period: RSI 계산에 사용할 기간 (기본값: 14)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(series, window=20, k=2):
    """
    볼린저 밴드 계산
    - series: 종가 데이터
    - window: 이동 평균 기간 (기본값: 20)
    - k: 표준편차 배수 (기본값: 2)
    """
    sma = series.rolling(window=window).mean()  # 이동평균
    std = series.rolling(window=window).std()  # 표준편차

    upper_band = sma + (k * std)  # 상단 밴드
    lower_band = sma - (k * std)  # 하단 밴드
    return sma, upper_band, lower_band


if __name__ == "__main__":
    data = fetch_data("KRW-XRP")
    data.to_csv("data/historical_data.csv")
