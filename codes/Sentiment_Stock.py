import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')
import ast
from datetime import datetime
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import yfinance as yf
import pandas as pd

# VIX와 S&P 500 데이터 다운로드
vix_data = yf.download("^VIX", start="2020-01-01", end="2024-11-14")
sp500_data = yf.download("^GSPC", start="2020-01-01", end="2024-11-14")

# VIX와 S&P 500의 종가 데이터프레임 병합
vix_sp500_df = pd.DataFrame({
    'Date': vix_data.index,
    'Close_VIX': vix_data['Close'],
    'Close_SP': sp500_data['Close'],
    'Volume_SP' : sp500_data['Volume']
}).dropna()

# NaN 값 제거
vix_sp500_df = vix_sp500_df.dropna()

sentiment = pd.read_excel('/Users/taekyounglee/Documents/projects/NH_Invest/codes/Cleaned_AAII_Sentiment_Survey.xlsx')
sentiment['Date'] = sentiment['Date'].astype('datetime64[ns]')

if 'Date' in vix_sp500_df.index.names:
    vix_sp500_df = vix_sp500_df.reset_index(drop=True)
    
merged_df = pd.merge(sentiment, vix_sp500_df, on = 'Date', how = 'inner')
analysis_df = merged_df[['Bullish', 'Neutral', 'Bearish', 'Close_VIX', 'Close_SP', 'Volume_SP']]


scaler = MinMaxScaler()
analysis_df = pd.DataFrame(scaler.fit_transform(analysis_df), columns = analysis_df.columns)

# 상관관계 분석
correlation_matrix = analysis_df.corr().round(2)

# 모든 셀에 값이 적히도록 히트맵을 녹색 느낌의 색상으로 생성
plt.figure(figsize=(7, 5))  # 그래프 크기 조정
sns.heatmap(correlation_matrix, cmap='YlGn', fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=16)
plt.show()

