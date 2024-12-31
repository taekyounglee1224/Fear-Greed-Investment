import pandas as pd
import yfinance as yf
import time
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')
import ast
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from itertools import product
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None  # default='warn'

crawling_df_merged = pd.read_csv('/Users/taekyounglee/Documents/projects/NH_Invest/data/preprocessed/result/stock/ETF크롤링결과취합.csv',encoding='cp949')

# -,-cash 추가 작업
new_rows = pd.DataFrame({
    'Ticker': ['-', '-CASH-'],
    'Open': [None, None],  # 필요에 따라 빈 값을 채워준다
    'High': [None, None],
    'Low': [None, None],
    'Close': [None, None],
    'Volume': [None, None]
})

# 두 개의 새 행을 데이터프레임의 맨 아래에 추가
crawling_df_merged_final = pd.concat([crawling_df_merged, new_rows], ignore_index=True)

# 구성종목파일 읽기
file_path = '/Users/taekyounglee/Documents/projects/NH_Invest/open_본선데이터/NH_CONTEST_DATA_ETF_HOLDINGS.csv' 
holding_df = pd.read_csv(file_path, encoding='cp949')

# 'etf_tck_cd'와 'tck_iem_cd'에서 공백 제거
holding_df['etf_tck_cd'] = holding_df['etf_tck_cd'].astype(str).str.strip()
holding_df['tck_iem_cd'] = holding_df['tck_iem_cd'].astype(str).str.strip()

# 'tck_iem_cd'에서 '/'을 '-'로 치환
holding_df['tck_iem_cd'] = holding_df['tck_iem_cd'].str.replace('/', '-')

# crawling_df_merged_final에서 'Ticker' 열의 값들 (공백 제거)
crawling_df_merged_final_values = crawling_df_merged_final['Ticker'].astype(str).str.strip().unique()

# 'etf_tck_cd'별로 'tck_iem_cd' 그룹핑
grouped_tck = holding_df.groupby('etf_tck_cd')['tck_iem_cd'].apply(list).reset_index()

# 각 etf_tck_cd에 해당하는 tck_iem_cd 값들이 모두 crawling_df_merged_final의 값들과 일치하는지 확인
df178 = grouped_tck[grouped_tck['tck_iem_cd'].apply(lambda x: all(item in crawling_df_merged_final_values for item in x))]

# 'tck_iem_cd'에서 고유값의 개수를 nunique로 확인
all_matching_tickers = [item for sublist in df178['tck_iem_cd'] for item in sublist]

# crawling_df_merged_final에서 matching_etfs의 'tck_iem_cd'에 해당하는 행을 필터링하여 새로운 데이터프레임 생성
dfEACH = crawling_df_merged_final[crawling_df_merged_final['Ticker'].isin(all_matching_tickers)]

# 날짜를 datetime 형식으로 변환
dfEACH.loc[:, 'Date'] = pd.to_datetime(dfEACH['Date'])

# 시작 날짜와 종료 날짜 설정 (datetime 형식으로 변환)
start_date = pd.to_datetime('2024-05-20')
end_date = pd.to_datetime('2024-08-27')

# 지정된 기간의 데이터만 필터링
df_filtered1 = dfEACH.loc[(dfEACH['Date'] >= start_date) & (dfEACH['Date'] <= end_date)].copy()

# 전날 대비 수익률 계산
df_filtered1.loc[:, 'Previous_Close'] = df_filtered1['Close'].shift(1)  # 전날의 종가
df_filtered1.loc[:, 'Return'] = (df_filtered1['Close'] - df_filtered1['Previous_Close']) / df_filtered1['Previous_Close'] * 100  # 수익률 계산

# 수익률이 양수면 'p', 음수면 'n' 추가
df_filtered1.loc[:, 'p/n'] = df_filtered1['Return'].apply(lambda x: 'p' if x >= 0 else 'n')

# 새로운 기간 설정 (datetime 형식으로 변환)
start_date = pd.to_datetime('2024-05-28')
end_date = pd.to_datetime('2024-08-26')

df_filtered2 = df_filtered1.loc[(df_filtered1['Date'] >= start_date) & (df_filtered1['Date'] <= end_date)].copy()

# tck_iem_cd 열을 리스트로 변환하고 '-' 값을 제거하는 함수 정의
def clean_tck_iem_cd(x):
    try:
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            cleaned_list = [i.strip() for i in ast.literal_eval(x) if i.strip() != '-']
            return cleaned_list if cleaned_list else ['-']
        else:
            return x
    except (ValueError, SyntaxError):
        return x

# 적용
df178.loc[:, 'tck_iem_cd'] = df178['tck_iem_cd'].apply(clean_tck_iem_cd)

# tck_iem_cd 열에서 '-CASH-'만 제거하는 함수 정의
def remove_cash_only(x):
    try:
        return [i for i in x if i.upper() != '-CASH-']
    except:
        return x

# 적용
df178.loc[:, 'tck_iem_cd'] = df178['tck_iem_cd'].apply(remove_cash_only)

# ETF별 긍정/부정 거래량 계산 함수 정의
def calculate_etf_volume_optimized(df_cleaned, df_volume):
    etf_volume_data = []

    df_volume_filtered = df_volume[['Date', 'Ticker', 'Volume', 'p/n']]

    for _, row in df_cleaned.iterrows():
        etf_ticker = row['etf_tck_cd']
        stock_tickers = row['tck_iem_cd']

        # ETF를 구성하는 모든 티커들의 데이터를 필터링
        stock_data = df_volume_filtered[df_volume_filtered['Ticker'].isin(stock_tickers)]

        # 날짜별로 긍정/부정 거래량 합산
        for date in stock_data['Date'].unique():
            day_data = stock_data[stock_data['Date'] == date]
            pos_volume_sum = day_data[day_data['p/n'] == 'p']['Volume'].sum()
            neg_volume_sum = day_data[day_data['p/n'] == 'n']['Volume'].sum()

            etf_volume_data.append({
                'etf_tck_cd': etf_ticker,
                'Date': date,
                'Positive_Volume': pos_volume_sum,
                'Negative_Volume': neg_volume_sum,
                'Pos-NegVolume': pos_volume_sum - neg_volume_sum
            })

    return pd.DataFrame(etf_volume_data)

# ETF별 긍정/부정 거래량 계산
final_PosNegVolume_df = calculate_etf_volume_optimized(df178, df_filtered2)

# 열 순서를 변경하고 날짜 형식을 설정
final_PosNegVolume_df = final_PosNegVolume_df[['Date', 'etf_tck_cd', 'Positive_Volume', 'Negative_Volume', 'Pos-NegVolume']]
final_PosNegVolume_df['Date'] = pd.to_datetime(final_PosNegVolume_df['Date']).dt.date


# 해외종목정보
stock_info_df = pd.read_csv('/Users/taekyounglee/Documents/projects/NH_Invest/open_본선데이터/NH_CONTEST_NW_FC_STK_IEM_IFO.csv',encoding='cp949')
# 해외종목정보에 있는 종목 일자별 시세 정보
price_info_df = pd.read_csv('/Users/taekyounglee/Documents/projects/NH_Invest/open_본선데이터/NH_CONTEST_STK_DT_QUT.csv',encoding='cp949')

# ETF의 일자별 시세 정보만 보기
etf_info_df = stock_info_df[stock_info_df['stk_etf_dit_cd'] == 'ETF']
price_info_df['tck_iem_cd'] = price_info_df['tck_iem_cd'].str.strip()

# 'ETF'에 해당하는 일자별 시세 정보만 병합
etf_sise_df = pd.merge(price_info_df, etf_info_df[['tck_iem_cd', 'stk_etf_dit_cd']], on='tck_iem_cd', how='inner')

# ETF 티커 추출
etf_tickers = etf_sise_df['tck_iem_cd'].unique()

print('Checkpoint 1')
# yfinance에서 15개월 간 종가 데이터 크롤링
start_date = '2023-05-27'
end_date = '2024-08-27'

all_etf_data = pd.DataFrame()
for ticker in etf_tickers:
    
    etf_data = yf.download(ticker, start=start_date, end=end_date,progress=False)
    
    etf_data['tck_iem_cd'] = ticker
    
    all_etf_data = pd.concat([all_etf_data, etf_data])

print('First Download Complete')
# yfinance 다운로드 종가 데이터 정리
print('Checkpoint 2')
if 'level_0' in all_etf_data.columns:
    all_etf_data.drop(columns=['level_0'], inplace=True)
all_etf_data.reset_index(drop=False, inplace=True)
yfinance_etf_3month_df = all_etf_data[['Date', 'Close', 'tck_iem_cd']]
yfinance_etf_3month_df['Date'] = pd.to_datetime(yfinance_etf_3month_df['Date']).dt.strftime('%Y%m%d')
yfinance_etf_3month_df.rename(columns={'Date': 'bse_dt'}, inplace=True)
yfinance_etf_3month_df.rename(columns={'Close': 'iem_end_pr'}, inplace=True)
yfinance_etf_3month_df = yfinance_etf_3month_df[['bse_dt', 'tck_iem_cd', 'iem_end_pr']]


# 날짜 데이터 처리
yfinance_etf_3month_df['bse_dt'] = pd.to_datetime(yfinance_etf_3month_df['bse_dt'], format='%Y%m%d')
momentum_dates = pd.date_range(start='2024-05-28', end='2024-08-26').strftime('%Y-%m-%d')  # 날짜 형식 변환
tickers = yfinance_etf_3month_df['tck_iem_cd'].unique()

# 데이터프레임 생성 (인덱스에 날짜, 열에 티커)
momentum_df = pd.DataFrame(index=momentum_dates, columns=tickers)

# 각 날짜별로 모멘텀 계산 및 저장
for date in momentum_dates:
    date_dt = pd.to_datetime(date)
    df_filtered_12m = yfinance_etf_3month_df[(yfinance_etf_3month_df['bse_dt'] <= date_dt) & (yfinance_etf_3month_df['bse_dt'] >= date_dt - pd.DateOffset(days=252))]
    
    # 12개월 Moving Average 계산
    moving_averages_12m = df_filtered_12m.groupby('tck_iem_cd')['iem_end_pr'].mean()
    
    # 해당 날짜의 종가 데이터 추출
    daily_prices = yfinance_etf_3month_df[yfinance_etf_3month_df['bse_dt'] == date_dt].set_index('tck_iem_cd')['iem_end_pr']
    
    # 모멘텀 계산 (12개월 이동 평균 - 해당 날짜의 종가)
    momentum = moving_averages_12m.reindex(tickers, fill_value=np.nan) - daily_prices.reindex(tickers, fill_value=np.nan)
    
    # 결과 저장
    momentum_df.loc[date] = momentum

# 인덱스를 열로 변환 (bse_dt 추가)
momentum_df.reset_index(inplace=True)
momentum_df.rename(columns={'index': 'bse_dt'}, inplace=True)

# 공휴일에 있는 nan값 데이터 제거
final_momentum_df = momentum_df.dropna(subset=['AAPB'])


#안전자산수요 계산을 위한 미국 국채 데이터 다운로드 및 정리
us_treasury_data = yf.download('^TNX', start=start_date, end=end_date,progress=False)

if 'level_0' in us_treasury_data.columns:
    us_treasury_data.drop(columns=['level_0'], inplace=True)
us_treasury_data.reset_index(drop=False, inplace=True)

selected_us_treasury_data = us_treasury_data[['Date', 'Close']]
selected_us_treasury_data['Date'] = pd.to_datetime(selected_us_treasury_data['Date']).dt.strftime('%Y%m%d')
selected_us_treasury_data.rename(columns={'Date': 'bse_dt'}, inplace=True)
selected_us_treasury_data.rename(columns={'Close': 'us_treasury'}, inplace=True)
selected_us_treasury_data = selected_us_treasury_data[['bse_dt', 'us_treasury']]



yfinance_etf_3month_df2 = yfinance_etf_3month_df.copy()  # 기존 데이터를 복사하여 작업

# 날짜 열을 datetime 형식으로 변환
yfinance_etf_3month_df2['bse_dt'] = pd.to_datetime(yfinance_etf_3month_df2['bse_dt'], format='%Y%m%d')

# ETF 별로 그룹화하여 20일 수익률을 계산
# groupby 이후 apply는 기존 인덱스를 유지하지 않을 수 있으므로 결과를 안전하게 재할당
yfinance_etf_3month_df2['etf_20day_return'] = yfinance_etf_3month_df2.groupby('tck_iem_cd')['iem_end_pr'].apply(
    lambda x: ((x - x.shift(20)) / x.shift(20)) * 100
).reset_index(level=0, drop=True)


# 이미 변환된 selected_columns_df2 사용 (Date를 'bse_dt', Close를 'us_treasury'로 변경한 데이터프레임)
selected_us_treasury_data2 = selected_us_treasury_data.copy()  # 기존 데이터를 복사하여 작업

# 날짜 열을 datetime 형식으로 변환
selected_us_treasury_data2['bse_dt'] = pd.to_datetime(selected_us_treasury_data2['bse_dt'], format='%Y%m%d')

# 20일 수익률을 계산 (shift를 사용하여 20일 전 값과 비교)
selected_us_treasury_data2['ustreasury_20day_return'] = (selected_us_treasury_data2['us_treasury'] - selected_us_treasury_data2['us_treasury'].shift(20)) / selected_us_treasury_data2['us_treasury'].shift(20) * 100


# bse_dt를 기준으로 두 데이터를 병합
final_SafeHavenDemand_df = pd.merge(yfinance_etf_3month_df2, selected_us_treasury_data2[['bse_dt', 'ustreasury_20day_return']], on='bse_dt', how='left')

# 각 ETF 자산의 20일 수익률에서 동일 날짜의 미국 국채 20일 수익률을 뺀 값 계산
final_SafeHavenDemand_df['safe_haven_demand'] = final_SafeHavenDemand_df['etf_20day_return'] - final_SafeHavenDemand_df['ustreasury_20day_return']
final_SafeHavenDemand_df = final_SafeHavenDemand_df.dropna()


etf_sise_df = etf_sise_df.rename(columns = {'Date' : 'bse_dt', 'etf_tck_cd' : 'tck_iem_cd'})
etf_sise_df['tck_iem_cd'] = etf_sise_df['tck_iem_cd'].str.replace(' ', '', regex=True)
etf_sise_df['Weighted_Buy'] = etf_sise_df['byn_cns_sum_qty'] * etf_sise_df['trd_cst']
etf_sise_df['Weighted_Sell'] = etf_sise_df['sll_cns_sum_qty'] * etf_sise_df['trd_cst']

# 거래대금 가중 매수/매도 비율
etf_sise_df['Weighted_Buy_Sell_Ratio'] = etf_sise_df['Weighted_Buy'] / etf_sise_df['Weighted_Sell']

etf_buy_sell = etf_sise_df[['bse_dt','tck_iem_cd','Weighted_Buy_Sell_Ratio']]
etf_buy_sell = etf_buy_sell.drop_duplicates(subset=['bse_dt', 'tck_iem_cd'], keep='last')

#buy_sell_df = etf_buy_sell.pivot(index='bse_dt', columns='tck_iem_cd', values='Weighted_Buy_Sell_Ratio')
buy_sell_df = etf_buy_sell.fillna(1.0)   #결측치는 1로 처리

final_BuySell_df=buy_sell_df


df178_list = df178['etf_tck_cd'].tolist()

### 49개 ETF에 포함된 것들만 추출 (변동성 지수 산출에 필요한 12일 전인 24-05-09부터 수집) (Markdown)
print('Checkpoint 3')
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

error_list = []
dataframes = []

for ticker in tqdm(df178_list, desc="Downloading", leave=False):
    try:
        data = yf.download(ticker, start='2024-05-09', end='2024-08-27', progress=False)
        if not data.empty:
            data.reset_index(inplace=True)
            data['Ticker'] = ticker
            dataframes.append(data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
        else:
            error_list.append(ticker)
    except Exception as e:
        error_list.append(ticker)
    
    time.sleep(0.1)

# 모든 성공적인 다운로드를 하나의 DataFrame으로 결합
if dataframes:
    final_df = pd.concat(dataframes)
    # final_df.set_index('Date', inplace=True)  
else:
    final_df = pd.DataFrame()

print('Second Download Complete')
### VVI-1 지수 산출 : (고가 - 저가)/거래량 (Markdown)
print('Checkpoint 4')
period = 5
filtered_market_data = final_df[['Date','Ticker', 'High', 'Low', 'Close', 'Volume']]
filtered_market_data['Volume'] = filtered_market_data['Volume'].replace(0, 1)

filtered_market_data['vvi'] = (filtered_market_data['High'] - filtered_market_data['Low'])/filtered_market_data['Volume'] #vvi = (고가 - 저가)/거래량
filtered_market_data['vvi_sum'] = filtered_market_data.groupby('Ticker')['vvi'].transform(lambda x: x.rolling(window=period).sum()) #5일 총합
filtered_market_data['vvi_ma'] = filtered_market_data.groupby('Ticker')['vvi_sum'].transform(lambda x: x.rolling(window=period).mean()) #5일 총합의 이동평균
filtered_market_data['vvi_std'] = filtered_market_data.groupby('Ticker')['vvi_sum'].transform(lambda x: x.rolling(window=period).std()) #5일 총합의 표준편차
filtered_market_data['vvi_z'] = (filtered_market_data['vvi_sum'] - filtered_market_data['vvi_ma'])/filtered_market_data['vvi_std'] #Z점수 변환

vvi_pivot = pd.pivot_table(filtered_market_data, values = 'vvi_z', index = 'Date', columns = 'Ticker')
vvi_pivot['MAKX'] = vvi_pivot['MAKX'].fillna(0)
vvi_pivot['QTR'] = vvi_pivot['QTR'].fillna(0)
vvi_pivot.reset_index(inplace=True)

filtered_market_data = vvi_pivot.melt(id_vars=['Date'], var_name='Ticker', value_name='vvi_z')

filtered_market_data_modified = filtered_market_data.dropna()
final_vvi1_df=filtered_market_data_modified


### VVI-2 지수 산출 : 종가의 표준편차 이용(Markdown)
period = 5
filtered_market_data = final_df[['Date','Ticker', 'High', 'Low', 'Close', 'Volume']]
filtered_market_data['Volume'] = filtered_market_data['Volume'].replace(0, 1e-2)

filtered_market_data['close_std'] = filtered_market_data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=period).std()) 
filtered_market_data['vvi2'] = filtered_market_data['close_std'] / filtered_market_data['Volume']
filtered_market_data['vvi2_sum'] = filtered_market_data.groupby('Ticker')['vvi2'].transform(lambda x: x.rolling(window=period).sum()) #5일 총합
filtered_market_data['vvi2_ma'] = filtered_market_data.groupby('Ticker')['vvi2_sum'].transform(lambda x: x.rolling(window=period).mean()) #5일 총합의 이동평균
filtered_market_data['vvi2_std'] = filtered_market_data.groupby('Ticker')['vvi2_sum'].transform(lambda x: x.rolling(window=period).std()) #5일 총합의 표준편차
filtered_market_data['vvi2_z'] = (filtered_market_data['vvi2_sum'] - filtered_market_data['vvi2_ma'])/filtered_market_data['vvi2_std'] #Z점수 변환


filtered_market_data_modified2 = filtered_market_data.dropna()

final_vvi2_df=filtered_market_data_modified2


selected_columns = ['bse_dt', 'tck_iem_cd', 'etf_20day_return', 'safe_haven_demand']
data_selected = final_SafeHavenDemand_df[selected_columns]

final_vvi1_df = final_vvi1_df.loc[:, final_vvi1_df.columns != 'Unnamed: 0']
final_vvi2_df = final_vvi2_df.loc[:, final_vvi2_df.columns != 'Unnamed: 0']
final_momentum_df.rename(columns={'Unnamed: 0': 'bse_dt'}, inplace=True)

# 각 지수들의 데이터프레임에서 bse_dt 형식 통일
final_SafeHavenDemand_df['bse_dt'] = pd.to_datetime(final_SafeHavenDemand_df['bse_dt'])
final_momentum_df['bse_dt'] = pd.to_datetime(final_momentum_df['bse_dt'])
final_BuySell_df['bse_dt'] = pd.to_datetime(final_BuySell_df['bse_dt'], format='%Y%m%d', errors='coerce')
final_vvi1_df['Date'] = pd.to_datetime(final_vvi1_df['Date'])
final_vvi2_df['Date'] = pd.to_datetime(final_vvi2_df['Date'])
final_PosNegVolume_df['Date'] = pd.to_datetime(final_PosNegVolume_df['Date'], format='%Y%m%d', errors='coerce')

final_vvi1_df = final_vvi1_df.rename(columns = {'Date' : 'bse_dt'})
final_vvi2_df = final_vvi2_df.rename(columns = {'Date' : 'bse_dt'})
final_PosNegVolume_df = final_PosNegVolume_df.rename(columns = {'Date' : 'bse_dt'})

final_vvi1_df = final_vvi1_df.rename(columns = {'Ticker' : 'tck_iem_cd'})
final_vvi2_df = final_vvi2_df.rename(columns = {'Ticker' : 'tck_iem_cd'})
final_PosNegVolume_df = final_PosNegVolume_df.rename(columns = {'etf_tck_cd' : 'tck_iem_cd'})


selected_columns = ['bse_dt', 'tck_iem_cd', 'etf_20day_return', 'safe_haven_demand']
safe_haven_selected = final_SafeHavenDemand_df[selected_columns]

# 각 지수 파일들을 long type으로 변환
final_momentum_df = final_momentum_df.melt(id_vars=['bse_dt'], var_name='tck_iem_cd', value_name='momentum')

# 각 산출된 지수dml 데이터프레임 병합
merged1_data = pd.merge(safe_haven_selected, final_momentum_df, on=['bse_dt', 'tck_iem_cd'], how='inner')
merged2_data = pd.merge(merged1_data, final_vvi1_df, on=['bse_dt', 'tck_iem_cd'], how='inner')
merged3_data = pd.merge(merged2_data, final_vvi2_df, on=['bse_dt', 'tck_iem_cd'], how='inner')
merged4_data = pd.merge(merged3_data, final_PosNegVolume_df, on=['bse_dt', 'tck_iem_cd'], how='inner')
merged5_data = pd.merge(merged4_data, final_BuySell_df, on=['bse_dt', 'tck_iem_cd'], how='inner')

required_tickers = [
    "AAPB", "AAPU", "AIBU", "AIRR", "AMDL", "AMDS", "AMZU", "AMZZ", "ARKG", "ARKK", "ARKQ", "ARKW", "AVMC",
    "AVUV", "BABX", "BBH", "CLDL", "CNRG", "CONL", "CURE", "CWB", "CWEB", "CWS", "DDIV", "DDM", "DIA", "DIG",
    "DIV", "DIVG", "DJD", "DJIA", "DOGG", "DPST", "DRN", "DUHP", "DUSL", "EDOW", "EQRR", "ERX", "EVX", "FBL",
    "FBT", "FDN", "FIW", "FNGG", "FPX", "FTGS", "FTXL", "FTXN", "FXN", "GGLL", "GRPM", "GURU", "GUSH", "HIBL",
    "ICVT", "IQQQ", "IVOO", "KBE", "KBWB", "KBWY", "KIE", "KLIP", "KORU", "KRE", "LTL", "MAKX", "MDY", "MDYG",
    "METU", "MGK", "MIDU", "MILN", "MSFL", "MSFU", "MVV", "NVD", "NVDL", "NVDU", "OIH", "OUSM", "PAVE", "PBE",
    "PEY", "PEZ", "PHO", "PILL", "PJP", "PSCE", "PSI", "PSR", "PTF", "PXE", "PXI", "QCLN", "QLD", "QQEW", "QQMG",
    "QQQ", "QQQA", "QQQE", "QQQM", "QQQU", "QQXT", "QRMI", "QTEC", "QTR", "QYLD", "QYLE", "QYLG", "RDOG", "RDVI",
    "RETL", "RND", "ROM", "RPG", "RSPC", "RSPH", "RSPN", "RSPR", "RSPT", "RXL", "RYLG", "SDVY", "SJNK", "SKYU",
    "SKYY", "SMH", "SMOT", "SOXL", "SOXQ", "SPHB", "SPHD", "SPHQ", "SPMD", "SPYG", "TCAF", "TDV", "TECL", "TMF",
    "TNA", "TPOR", "TQQQ", "TSDD", "TSL", "TSLL", "TSLR", "UDOW", "UMDD", "UPW", "URE", "USD", "UTSL", "UXI",
    "UYM", "VNQ", "VOOG", "WANT", "WCLD", "WEBL", "WFH", "XES", "XHB", "XHE", "XHS", "XITK", "XLC", "XLU", "XLV",
    "XLY", "XMHQ", "XMMO", "XRT", "XSD", "XSHD", "XSHQ", "XTN", "YINN"
]

# merged_data에서 tck_iem_cd 값이 required_tickers에 속하는 데이터만 필터링
filtered_merged_data = merged5_data[merged5_data['tck_iem_cd'].isin(required_tickers)]
filtered_merged_data['bse_dt'] = pd.to_datetime(filtered_merged_data['bse_dt'], format='%Y%m%d').dt.strftime('%Y-%m-%d')


endprice_data = pd.DataFrame(columns=['bse_dt', 'tck_iem_cd', 'Close'])

for ticker in required_tickers:
    data = yf.download(ticker, start="2024-05-28", end="2024-08-26",progress=False
)
    if not data.empty:
        df = pd.DataFrame({
            'bse_dt': data.index,
            'tck_iem_cd': ticker,
            'Close': data['Close']
        })

        df = df.dropna()
        endprice_data = pd.concat([endprice_data, df])

filtered_merged_data['bse_dt'] = pd.to_datetime(filtered_merged_data['bse_dt'])
filtered_merged_data['tck_iem_cd'] = filtered_merged_data['tck_iem_cd'].astype(str)
endprice_data['tck_iem_cd'] = endprice_data['tck_iem_cd'].astype(str)

# 종가 병합
final_merged_data = pd.merge(filtered_merged_data, endprice_data, on=['bse_dt', 'tck_iem_cd'], how='inner')


#매수/매도 기준치 설정
sell_critical = 67
buy_critical = 33

#거래 수수료
transaction_fee = 0.0025
fixed_fee = 0.01
SEC_fee = 0.0000278

# 가중치 설정: 0.1 단위로, 각 가중치는 0도 가능
possible_weights = [i / 20 for i in range(21)]  # 0부터 1.0까지 0.1 단위
weight_combinations = [comb for comb in product(possible_weights, repeat=5) if np.isclose(sum(comb), 1.0)]

# 변수 초기화
best_weights = None
best_avg_return = -np.inf
best_total_return = -np.inf
best_index_profit = 0
best_better_count = 0


scaler = MinMaxScaler()

# 스케일링을 적용할 컬럼들
scale_columns = ['momentum', 'Pos-NegVolume', 'Weighted_Buy_Sell_Ratio', 'safe_haven_demand', 'vvi_z', 'vvi2_z']

for col in scale_columns:
    # 컬럼을 수치형으로 변환
    final_merged_data[col] = pd.to_numeric(final_merged_data[col], errors='coerce')
    
    # NaN이나 무한대 값을 갖는 경우, 해당 그룹의 평균으로 대체 (평균이 무한대일 경우 0으로 대체)
    final_merged_data[col] = final_merged_data.groupby('tck_iem_cd')[col].transform(lambda x: np.where(np.isfinite(x), x, np.nan))
    mean_value = final_merged_data.groupby('tck_iem_cd')[col].transform('mean')
    mean_value = np.where(np.isfinite(mean_value), mean_value, 0)  # 평균이 무한대인 경우를 대비
    
    # mean_value를 Series로 변환하여 fillna에 사용
    mean_series = pd.Series(mean_value, index=final_merged_data.index)
    final_merged_data[col].fillna(mean_series, inplace=True)

# 비스케일 컬럼 선택
non_scale_columns = final_merged_data.drop(columns=scale_columns)

# 각 'tck_iem_cd' 그룹별로 스케일링 적용
scaled_data = []
for name, group in final_merged_data.groupby('tck_iem_cd'):
    group[scale_columns] = scaler.fit_transform(group[scale_columns])
    scaled_data.append(group)

total_index = pd.concat(scaled_data, axis=0)

total_index = pd.merge(non_scale_columns, total_index, left_index=True, right_index=True, suffixes=('', '_drop'))
total_index.drop(total_index.filter(regex='_drop$').columns.tolist(), axis=1, inplace=True)

# 조건에 따라 일부 지표는 1 - minmax로 변환
total_index['safe_haven_demand'] = 1 - total_index['safe_haven_demand']
total_index['vvi_z'] = 1 - total_index['vvi_z']
total_index['vvi2_z'] = 1 - total_index['vvi2_z']


print('Checkpoint 5')
from joblib import Parallel, delayed

nominal_interest_rate = 0.0475  #실제 명목 금리
effective_interest_rate = (1 + nominal_interest_rate / 12) ** 3 - 1 #3개월 유효 금리 (월복리 가정)
print(np.round(effective_interest_rate * 100, 3),"%")

def cumulative_roi_rate(df):
    total_return = 0
    count = 0
    
    for stock, group in df.groupby('tck_iem_cd'):
        last_return = group['rate_of_return'].iloc[-1]  # 각 그룹의 마지막 수익률
        
        if group['index>original_count'].iloc[-1] == 1:  # 금리보다 좋은 경우만 수익률 계산
            total_return += last_return
            count += 1
        else:
            continue
    
    if count > 0:
        average_return = (total_return / count) * 100
        total_return_percentage = np.round(total_return * 100, 2)
        average_return_percentage = np.round(average_return, 2)
    else:
        total_return_percentage = 0
        average_return_percentage = 0
    
    return total_return_percentage, average_return_percentage
    

def evaluate_weights(weights):
    weight_momentum, weight_mcllelan, weight_putcall, weight_safehaven, weight_vvi_total = weights

    total_index['fgi_index'] = (
        total_index['momentum'] * weight_momentum +
        total_index['Pos-NegVolume'] * weight_mcllelan +
        total_index['Weighted_Buy_Sell_Ratio'] * weight_putcall +
        total_index['safe_haven_demand'] * weight_safehaven +
        (total_index['vvi_z'] * 0.5 + total_index['vvi2_z'] * 0.5) * weight_vvi_total
    ) * 100

    total_index['buy_sell'] = 0
    for stock, group in total_index.groupby('tck_iem_cd'):
        state = 0
        for i in group.index[:-1]:
            if state == 0 and total_index.at[i, 'fgi_index'] <= buy_critical:
                total_index.at[i, 'buy_sell'] = -1
                state = -1
            elif state == -1 and total_index.at[i, 'fgi_index'] >= sell_critical:
                total_index.at[i, 'buy_sell'] = 1
                state = 1
            elif state == 1 and total_index.at[i, 'fgi_index'] <= buy_critical:
                total_index.at[i, 'buy_sell'] = -1
                state = -1
            else:
                total_index.at[i, 'buy_sell'] = 0
        last_row = group.index[-1]
        if state == -1:
            total_index.at[last_row, 'buy_sell'] = 1
        else:
            total_index.at[last_row, 'buy_sell'] = 0

    total_index['cost_income'] = np.where(
        total_index['buy_sell'] != 0,
        total_index['Close_x'] * (total_index['buy_sell'] - transaction_fee - SEC_fee) - fixed_fee, 0
    )

    for stock, group in total_index.groupby('tck_iem_cd'):
        cost_income_sum = group['cost_income'].sum()
        total_index.loc[group.index[-1], 'revenue_index'] = cost_income_sum

    for stock, group in total_index.groupby('tck_iem_cd'):
        if (group['buy_sell'] == -1).any():
            rate_of_return = group['revenue_index'].iloc[-1] / group[group['buy_sell'] == -1]['Close_x'].iloc[0]
        else:
            rate_of_return = float('nan')
        total_index.loc[group.index[-1], 'rate_of_return'] = rate_of_return

    total_index['index>original_count'] = np.where(total_index['rate_of_return'] > effective_interest_rate, 1, 0)

    total_return_percentage, average_return_percentage = cumulative_roi_rate(total_index)
    index_profit = (total_index['revenue_index'] * total_index['index>original_count']).sum()
    better_count = total_index['index>original_count'].sum()

    return weights, average_return_percentage, total_return_percentage, index_profit, better_count

# 병렬 처리 수행
results = Parallel(n_jobs=-1)(delayed(evaluate_weights)(weights) for weights in weight_combinations)

# 최적 가중치 찾기
best_avg_return = -np.inf
for weights, avg_return, total_return, profit, count in results:
    if avg_return > best_avg_return:
        best_avg_return = avg_return
        best_weights = weights
        best_total_return = total_return
        best_index_profit = profit
        best_better_count = count

# 최적 가중치와 결과 출력
print("최적 가중치:", best_weights)
print("최적 평균 수익률:", best_avg_return, "%")
print("총 수익률:", best_total_return, "%")
print("지수를 이용해서 창출한 총수익:", best_index_profit)
print("지수로 수익 창출했을 때 더 좋게 나오는 종목 수:", best_better_count, "/ 178")
