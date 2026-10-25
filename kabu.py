import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# 警告を非表示にする
warnings.simplefilter('ignore')

# 画面のタイトル
st.title("📈 株式将来予測アプリ (Prophet)")
st.write("銘柄コードや予測期間を指定して、将来の株価を予測します。")

# サイドバー（画面の左側）に入力欄を作る
st.sidebar.header("予測条件の設定")

# 1. 銘柄コードの入力
ticker_input = st.sidebar.text_input("銘柄コード（例: 4755.T, AAPL）", value="4755.T")
ticker_symbol = ticker_input.strip()

# 数字だけの場合は自動で '.T' を追加する
if '.' not in ticker_symbol and ticker_symbol.isdigit():
    ticker_symbol = ticker_symbol + '.T'

# 2. 予測期間（日数）の入力
forecast_days = st.sidebar.slider("何日先まで予測しますか？", min_value=10, max_value=180, value=30)

# 3. 終了日の入力
default_end = datetime.today().date()
end_date = st.sidebar.date_input("データの終了日", value=default_end)

start_date = '2025-01-01'

# 実行ボタン
if st.sidebar.button("予測を実行する"):
    with st.spinner(f"{ticker_symbol} のデータを取得して予測中..."):
        try:
            # データの取得
            stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
            
            if stock_data.empty:
                st.error("データが取得できませんでした。銘柄コードを確認してください。")
            else:
                # 終値データを安全に抽出
                if isinstance(stock_data.columns, pd.MultiIndex):
                    if ('Close', ticker_symbol) in stock_data.columns:
                        close_price = stock_data[('Close', ticker_symbol)]
                    else:
                        close_price = stock_data.iloc[:, 0]
                else:
                    if 'Close' in stock_data.columns:
                        close_price = stock_data['Close']
                    else:
                        close_price = stock_data.iloc[:, 0]

                # Prophet用にデータフレームを整形
                df = pd.DataFrame({'ds': close_price.index, 'y': close_price.values})
                df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

                # モデルの作成と学習
                model = Prophet()
                model.fit(df)

                # 将来の予測データを作成
                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)

                # グラフを描画してStreamlitに表示
                fig, ax = plt.subplots(figsize=(10, 5))
                model.plot(forecast, ax=ax)
                ax.set_title(f'{ticker_symbol} Stock Price Prediction ({forecast_days} Days)', fontsize=14)
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.grid(True)
                
                st.pyplot(fig)
                st.success("予測が完了しました！")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
