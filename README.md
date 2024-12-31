## 2024 NH Investment Securities Competition - US ETF Investment Strategy Using Generative AI: Based on the Fear and Greed Index

### 0. Introduction

Investors often make investment decisions based on personal emotions and psychology, and are prone to irrational behavior when extreme emotions such as fear and greed are expressed in the market. For example, when the market is gripped by greed, the market has the potential to be overvalued, and when fear takes over, the risk of being undervalued increases. These emotional swings affect how investors decide when to exit or enter the market. Therefore, investors need to be able to effectively manage market fluctuations by not letting their emotions get the best of them and maintaining their long-term investment goals, which is where the fear-greed investment strategy comes in.


<img width="204" alt="image" src="https://github.com/user-attachments/assets/90e2aa80-06d6-43cb-9007-f3d3579f2fe2" />

### I. 공포-탐욕지수란?

The Fear & Greed Index, published by CNN, is a measure of stock sentiment
CNN's Fear & Greed Index quantifies the emotional tone of the market by measuring the state of mind of investors and ranges from 0 to 100. A reading closer to 100 indicates extreme greed, while a reading closer to 0 indicates extreme fear. The strategy is to buy when the index is low and sell when it is high, following the saying “buy on fear, sell on greed”.

- 1. Market Momentum: Moving average of the previous 125 trading days
- 2. Stock Price Strength: The ratio of the number of stocks with new 52-week highs to the number of stocks with new 52-week lows on the New York Stock Exchange.
- 3. Stock Price Breadth: The difference between the volume of rising stocks and falling stocks.
- 4. Put and Call Options Ratio: The ratio of put-options to call-options.
- 5. Market Volatility: Measures the volatility of the market utilizing the VIX index.
- 6. Safe Haven Demand: The difference between U.S. Treasury and equity yields.
- 7. Junk Bond Demand: The spread between junk bond and Treasury yields.


### II. Our Fear-Greed Index

1. Momentum (Higher = Greed, Lower = Fear)
   
- Original CNN Fear-Greed Index Factor: Market Momentum
- Our Method: Measured by comparing the closing price of a specific ETF to its moving average over the previous 12 months (252 trading days).
- Variation: The existing CNN Index measures market-wide momentum using a 125 trading day moving average of the S&P 500 Index. We used a 252 trading day moving average to assess market momentum for individual ETFs
- Reference : Time series momentum - Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012)

2. demand for safe-haven assets (Higher = Fear, Lower = Greed)
   
- Original CNN Fear-Greed Index Factor: Safe Haven Demand
- Our method: Calculated using the difference between the 1-month (20-trading day) return of a specific ETF and the 1-month (20-trading day) return of the U.S. Treasury
- Variation: The CNN Fear-Greed Index uses the yield differential between U.S. Treasuries and stocks to assess market-wide safe haven appetite. While applying the same approach to reflect market conditions for individual ETF assets, we determined that calculating returns based on 20 trading days better reflects the market risk of individual assets
- Reference : VKOSPI 지수를 이용한 단기 주가수익률 예측에 관한 연구 : 이정환, 손삼호, 이건희 (2024)

3. Positive-Negative Trade (Higher = Greed, Lower = Fear)
   
- Original CNN Fear-Greed Index Factor: Stock Price Breadth
- Our method: Calculates the difference between the positive and negative trading volume of the individual equity holdings that make up an ETF
- Variation: The CNN index compares positive and negative volume across the market as a whole. Given that ETF assets are composed of individual stocks and derivatives, we sum positive and negative trading volume only for the stock assets included in the ETF. This variation was necessary for a more accurate volume-based sentiment analysis because derivatives do not reflect the direct price movement of stocks.
- Reference : 김근택. (2021). 레버리지, 인버스ETF의 개인 매수/매도 데이터의 투자심리지수 유용성과 활용성

4. Bid/Ask Price Ratio (Higher = Greed, Lower = Fear)
   
- Original CNN Fear-Greed Index Factor: Put and Call Options Ratio (Put and Call Options)
- Our method: Uses a ratio that takes into account the number of trades executed and the price of the trades.
- Variation: The CNN index uses options market data, but our data did not have options information. Instead, we utilized the number of bids and offers and trade sizes to reflect market sentiment. We include trade size because large capital flows may be more important. We believe this approach can reflect market sentiment similarly to the put/call ratio.
- Reference : 투자 주체별 거래행태의 특징이 주식시장 수익률에 미치는 영향 - 백기태*・민병길**・백지원***

5. Volatility vs. Volume Index (Higher = Fear, Lower = Greed)
   
- Existing CNN Fear-Greed Index Factor: Market Volatility
- Current method: Combining (high-low)/volume with the standard deviation of the closing price/volume metric
- Variation: The CNN Index uses the VIX to measure market volatility. We calculate the volatility of individual ETF assets by correlating it to their trading volume. This reflects the fact that the relationship between volatility and volume is closely linked to investor sentiment. The rationale for the volatility-volume index is based on established research that shows the combination of volatility and volume in the market acts as a psychological signal
  

### III. Methodology
#### 1) Index Scaling
For each of the five metrics we calculated, we applied the min-max scaling method to normalize each of them to a value between 0 and 1, and then multiplied it by 100 to give it a value between 0 and 100. We converted them into indicators and calculated the Fear-Greed Index, which has a value between 0 and 100, through a weighted sum

출처 : https://edition.cnn.com/markets/fear-and-greed



#### 2) Weight Parameter
Methodology: We optimized the weights for each metric to maximize the average return by setting the weight between 0 and 1 and the unit to 0.05.

$$ max Z=ARR\left(\mathrm{average\ rate\ of\ return}\right)$$
$$s.t.\;\;w_1+w_2+w_3+w_4+w_5=1$$
    $$w_i=range\left(0,\ 1,\ 0.05\right)\emsp\left(1\le i\le5\right)$$


### IV. Tools

- Python 3.11
- Microsoft Azure
- Tableau

### V. References

- Time series momentum - Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012)
- VKOSPI 지수를 이용한 단기 주가수익률 예측에 관한 연구 : 이정환, 손삼호, 이건희 (2024)
- 레버리지, 인버스ETF의 개인 매수/매도 데이터의 투자심리지수 유용성과 활용성 : 김근택. (2021)
- 투자 주체별 거래행태의 특징이 주식시장 수익률에 미치는 영향 - 백기태*・민병길**・백지원***
- https://edition.cnn.com/markets/fear-and-greed





















