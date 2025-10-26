import os
from dotenv import load_dotenv
load_dotenv()
import pyupbit

def execute_trade(ticker, signal):
    """
    매매 신호를 기반으로 매매를 자동으로 실행
    """

    api_key = os.getenv("ACCESS_KEY")
    secret_key = os.getenv("SECRET_KEY")
    upbit = pyupbit.Upbit(api_key, secret_key)
    
    if signal == "buy":
        balance = upbit.get_balance("KRW")
        buy_amount = balance * 0.9995
        if buy_amount > 5000: 
            upbit.buy_market_order(ticker, buy_amount)
            current_price = pyupbit.get_current_price(ticker)
            purchased_quantity = buy_amount / current_price
            print(f"매수 완료: {ticker}, 수량: {purchased_quantity:.6f} (가격: {current_price:.2f}원)")

    elif signal == "sell":
        balance = upbit.get_balance(ticker)
        if balance > 0:
            upbit.sell_market_order(ticker, balance)
            current_price = pyupbit.get_current_price(ticker)
            print(f"매도 완료: {ticker}, 수량: {balance:.6f} (가격: {current_price:.2f}원)")

if __name__ == "__main__":
    execute_trade("KRW-XRP", "hold")