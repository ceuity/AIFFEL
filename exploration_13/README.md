# 13. 어제 오른 내 주식, 과연 내일은?

오늘은 시계열 예측(Time-Series Prediction)을 다루는 여러 가지 통계적 기법 중에 가장 널리 알려진 ARIMA(Auto-regressive Integrated Moving Average)에 대해 알아보고 이를 토대로 특정 주식 종목의 가격을 예측해 보는 실습을 진행해보자.

## 학습 목표

---

- 시계열 데이터의 특성과 안정적(Stationary) 시계열의 개념을 이해한다.
- ARIMA 모델을 구성하는 AR, MA, Diffencing의 개념을 이해하고 간단한 시계열 데이터에 적용해 본다.
- 실제 주식 데이터에 ARIMA를 적용해서 예측 정확도를 확인해 본다.

## 시계열 예측

### (1) 미래를 예측한다는 것은 가능할까?

---

- 지금까지의 주가변곡선을 바탕으로 다음 주가변동 예측
- 특정 지역의 기후데이터를 바탕으로 내일의 온도변화 예측
- 공장 센터데이터 변화이력을 토대로 이상 발생 예측

위 예시의 공통점은 예측 근거가 되는 시계열(Time-Series) 데이터가 있다는 것이다. 시계열 데이터란 시간 순서대로 발생한 데이터의 수열이라는 뜻이다.

$$ Y = \lbrace Y_t: t ∈ T \rbrace \text{, where }T\text{ is the index set}  $$

일정 시간 간격으로 발생한 데이터 뿐만 아니라 매일의 주식 거래 가격을 날짜-가격 형태로 날짜순으로 모아둔 데이터가 있다면 이 데이터도 마찬가지로 훌륭한 시계열 데이터가 될 것이다.

그렇다면 특정 주식의 매일 가격 변동 시계열 데이터가 수년 치 쌓여있다고 할 때, 이 데이터를 토대로 내일의 주식 가격이 얼마가 될지, 오를지 내릴지를 예측할 수 있을까? 결론적으로 말하자면 미래 예측은 불가능한 것이다. 그럼에도 불구하고 미래의 데이터를 예측하려고 한다면 두 가지 전제가 필요하다.

- 과거의 데이터에 일정한 패턴이 발견된다
- 과거의 패턴은 미래에도 동일하게 반복될 것이다.

이 두 가지 문장이 의미하는 바는 즉, 안정적(Stationary)인 데이터에 대해서만 미래 예측이 가능하다는 것이다. 여기서 안정적(Stationary)이다는 것은 시계열 데이터의 통계적 틍성이 변하지 않는다는 뜻이다.

### (2) Stationary한 시계열 데이터

---

1. 시간의 추이와 관계 없이 평균이 불변

    ![images00.png](./images00.png)

2. 시간의 추이와 관계 없이 분산이 불변

    ![images01.png](./images01.png)

3. 두 시점 간의 공분산이 기준 시점과 무관

    ![images02.png](./images02.png)

## 시계열 데이터 사례분석

### (1) Daily Minimum Temperatures in Melbourne

---

**데이터 준비**

```jsx
$ wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv
$ wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
```

**시계열(Time Series) 생성**

첫 번째로 다루어볼 데이터는 Daily Minimum Temperatures in Melbourne이다.

```python
# 모듈 import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# data load
dataset_filepath = os.getenv('HOME')+'/aiffel/stock_prediction/data/daily-min-temperatures.csv' 
df = pd.read_csv(dataset_filepath) 
print(type(df))
df.head()

# 이번에는 Date를 index_col로 지정 
df = pd.read_csv(dataset_filepath, index_col='Date', parse_dates=True)
print(type(df))
df.head()

ts1 = df['Temp']
print(type(ts1))
ts1.head()

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 13, 6 

# 시계열(time series) 데이터를 시각화. 특별히 더 가공하지 않아도 잘 그려진다.
plt.plot(ts1)

# 시계열(Time Series)에서 결측치가 있는 부분만 Series로 출력
ts1[ts1.isna()]  

# 결측치가 있다면 이를 보간합니다. 보간 기준은 time을 선택합니다. 
ts1=ts1.interpolate(method='time')

# 보간 이후 결측치(NaN) 유무를 다시 확인합니다.
print(ts1[ts1.isna()])

# 다시 그래프를 확인해봅시다!
plt.plot(ts1)
```

![images03.png](./images03.png)

```python
# 일정 구간 내 통계치(Rolling Statistics)를 시각화
def plot_rolling_statistics(timeseries, window=12):
    
    rolmean = timeseries.rolling(window=window).mean()  # 이동평균 시계열
    rolstd = timeseries.rolling(window=window).std()    # 이동표준편차 시계열

     # 원본시계열, 이동평균, 이동표준편차를 plot으로 시각화해 본다.
    orig = plt.plot(timeseries, color='blue',label='Original')    
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

plot_rolling_statistics(ts1, window=12)
```

![images04.png](./images04.png)

시간에 따라 일정한 평균, 분산, 자기공분산의 패턴이 나타나는 것 처럼 보이므로 안정적인 시계열 데이터로 볼 수 있을 것이다. 좀 더 명확하게 하려면 통계적 접근이 필요하다.

### (2) International airline passengers

---

```python
# data load
dataset_filepath = os.getenv('HOME')+'/aiffel/stock_prediction/data/airline-passengers.csv' 
df = pd.read_csv(dataset_filepath, index_col='Month', parse_dates=True).fillna(0)  
print(type(df))
df.head()

ts2 = df['Passengers']
plt.plot(ts2)

plot_rolling_statistics(ts2, window=12)
```

![images05.png](./images05.png)

위의 사례와는 달리 시간의 추이에 따라 평균과 분산이 증가하는 패턴을 보인다면 이 시계열 데이터는 적어도 안정적이진 않다고 정성적인 결론을 내려볼 수 있을 것 같다. 이런 불안정적(Non-Stationary) 시계열 데이터에 대한 시계열 분석 기법도 알아보자.

## Stationary 여부를 체크하는 통계적 방법

### (1) Augmented Dickey-Fuller Test

---

Augmented Dickey-Fuller Test(ADF Test)라는 시계열 데이터의 안정성을 테스트하는 통계적 방법을 소개한다. 테스트는 주어진 시계열 데이터가 안정적이지 않다라는 귀무가설(Null Hypothesis)를 세운 후, 통계적 가설 검정 과정을 통해 이 귀무가설이 기각될 경우에 이 시계열 데이터가 안정적이다라는 대립가설(Alternative Hypothesis)을 채택한다는 내용이다.

통계적 가설 검정의 기본 개념을 이루는 p-value 등의 용어에 대해서는 한 번쯤 짚고 넘어가는 것이 좋을 것이다.

### (2) statsmodels 패키지와 adfuller 메소드

---

`statsmodels` 패키지는 R에서 제공하는 통계검정, 시계열분석 등의 기능을 파이썬에서도 이용할 수 있도록 하는 강력한 통계 패키지입니다. 이번 노드에서는 `statsmodels` 패키지의 기능을 자주 활용하게 될 것입니다. 아래는 `statsmodels` 패키지에서 제공하는 `adfuller` 메소드를 이용해 주어진 timeseries에 대한 Augmented Dickey-Fuller Test를 수행하는 코드이다.

```python
from statsmodels.tsa.stattools import adfuller

def augmented_dickey_fuller_test(timeseries):
    # statsmodels 패키지에서 제공하는 adfuller 메소드를 호출합니다.
    dftest = adfuller(timeseries, autolag='AIC')  
    
    # adfuller 메소드가 리턴한 결과를 정리하여 출력합니다.
    print('Results of Dickey-Fuller Test:')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

# Daily Minimum Temperatures in Melbourne
augmented_dickey_fuller_test(ts1)

"""
Results of Dickey-Fuller Test:
Test Statistic                   -4.444805
p-value                           0.000247
#Lags Used                       20.000000
Number of Observations Used    3629.000000
Critical Value (1%)              -3.432153
Critical Value (5%)              -2.862337
Critical Value (10%)             -2.567194
dtype: float64
"""
```

`Daily Minimum Temperatures in Melbourne` 시계열이 안정적이지 않다는 귀무가설은 p-value가 거의 0에 가깝게 나타났다. 따라서 이 귀무가설은 기각되고, 이 시계열은 안정적 시계열이라는 대립가설이 채택된다.

```python
# International airline passengers
augmented_dickey_fuller_test(ts2)

"""
Results of Dickey-Fuller Test:
Test Statistic                   0.815369
p-value                          0.991880
#Lags Used                      13.000000
Number of Observations Used    130.000000
Critical Value (1%)             -3.481682
Critical Value (5%)             -2.884042
Critical Value (10%)            -2.578770
dtype: float64
"""
```

`International airline passengers` 시계열이 안정적이지 않다는 귀무가설은 p-value가 거의 1에 가깝게 나타났다. 이것이 바로 이 귀무가설이 옳다는 직접적인 증거가 되지는 않지만, 적어도 이 귀무가설을 기각할 수는 없게 되었으므로 이 시계열이 안정적인 시계열이라고 말할 수는 없다.

## 시계열 예측의 기본 아이디어: Stationary 하게 만들기

---

안정적이지 않은 시계열을 안정적인 시계열로 바꾸기 위해 크게 두 가지 방법을 사용할 것이다. 한가지는 정성적인 분석을 통해 보다 안정적(starionary)인 특성을 가지도록 기존의 시계열 데이터를 가공/변형하는 시도들이고, 다른 하나는 시계열 분해(Time series decomposition)라는 기법을 적용하는 것이다.

### (1) 보다 Stationary한 시계열로 가공하기

---

**로그함수 변환**

```python
ts_log = np.log(ts2)
plt.plot(ts_log)

augmented_dickey_fuller_test(ts_log)

"""
Results of Dickey-Fuller Test:
Test Statistic                  -1.717017
p-value                          0.422367
#Lags Used                      13.000000
Number of Observations Used    130.000000
Critical Value (1%)             -3.481682
Critical Value (5%)             -2.884042
Critical Value (10%)            -2.578770
dtype: float64
"""
```

p-value가 0.42로 무려 절반 이상 줄어들었다. 정성적으로도 시간 추이에 따른 분산이 일정해진 것을 확인할 수 있다. 아주 효과적인 변환이었던 것 같이 보이나, 가장 두드러지는 문제점은 시간 추이에 따라 평균이 계속 증가한다는 점이다.

**Moving average 제거 - 추세(Trend) 상쇄하기**

시계열 분석에서 위와 같이 시간 추이에 따라 나타나는 평균값 변화를 추세(trend)라고 한다. 이 변화량을 제거해 주려면 거꾸로 Moving Average, 즉 rolling mean을 구해서 `ts_log`를 빼주면 된다.

```python
# moving average구하기 
moving_avg = ts_log.rolling(window=12).mean()  
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

# 변화량 제거
ts_log_moving_avg = ts_log - moving_avg 
ts_log_moving_avg.head(15)

# 결측치 제거
ts_log_moving_avg.dropna(inplace=True)
ts_log_moving_avg.head(15)

plot_rolling_statistics(ts_log_moving_avg)

augmented_dickey_fuller_test(ts_log_moving_avg)

"""
Results of Dickey-Fuller Test:
Test Statistic                  -3.162908
p-value                          0.022235
#Lags Used                      13.000000
Number of Observations Used    119.000000
Critical Value (1%)             -3.486535
Critical Value (5%)             -2.886151
Critical Value (10%)            -2.579896
dtype: float64
"""
```

![images06.png](./images06.png)

p-value가 0.02 수준이 되었으므로, 95% 이상의 confidence로 이 time series는 stationary하다고 할 수 있을 것이다. 그러나 지금까지의 접근에서 한가지 숨겨진 문제점이 있다. 바로 Moving Average를 계산하는 window=12로 정확하게 지정해 주어야 한다는 점이다. 만약 위 코드에서 window=6을 적용하면 다음과 같은 결과가 나온다.

```python
moving_avg_6 = ts_log.rolling(window=6).mean()
ts_log_moving_avg_6 = ts_log - moving_avg_6
ts_log_moving_avg_6.dropna(inplace=True)

plot_rolling_statistics(ts_log_moving_avg_6)

augmented_dickey_fuller_test(ts_log_moving_avg_6)

"""
Results of Dickey-Fuller Test:
Test Statistic                  -2.273822
p-value                          0.180550
#Lags Used                      14.000000
Number of Observations Used    124.000000
Critical Value (1%)             -3.484220
Critical Value (5%)             -2.885145
Critical Value (10%)            -2.579359
dtype: float64
"""
```

![images07.png](./images07.png)

그래프를 정성적으로 분석해서는 window=12일 때와 별 차이를 느낄수 없지만 Augmented Dickey-Fuller Test의 결과 p-value는 0.18 수준이어서 아직도 안정적 시계열이라고 말할 수 없게 되었다.

이 데이터셋은 월 단위로 발생하는 시계열이므로 12개월 단위로 주기성이 있기 때문에 window=12가 적당하다는 것을 추측할 수도 있을 것 같습니다만, moving average를 고려할 때는 rolling mean을 구하기 위한 window 크기를 결정하는 것이 매우 중요하다는 것을 기억해두자.

**차분(Differencing) - 계절성(Seasonality) 상쇄하기**

Trend에는 잡히지 않지만 시계열 데이터 안에 포함된 패턴이 파악되지 않은 주기적 변화는 예측에 방해가 되는 불안정성 요소이다. 이것은 Moving Average 제거로는 상쇄되지 않는 효과로, 이런 계절적, 주기적 패턴을 계절성(Seasonality)라고 한다.

이런 패턴을 상쇄하기 위해 효과적인 방법에는 차분(Differencing)이 있다. 시계열을 한 스텝 앞으로 시프트한 시계열을 원래 시계열에 빼 주는 방법이다.  이렇게 되면 남은 것은 현재 스텝 값 - 직전 스텝 값이 되어 정확히 이번 스텝에서 발생한 변화량을 의미하게 된다.

```python
ts_log_moving_avg_shift = ts_log_moving_avg.shift()

plt.plot(ts_log_moving_avg, color='blue')
plt.plot(ts_log_moving_avg_shift, color='green')

ts_log_moving_avg_diff = ts_log_moving_avg - ts_log_moving_avg_shift
ts_log_moving_avg_diff.dropna(inplace=True)
plt.plot(ts_log_moving_avg_diff)

plot_rolling_statistics(ts_log_moving_avg_diff)

augmented_dickey_fuller_test(ts_log_moving_avg_diff)

"""
Results of Dickey-Fuller Test:
Test Statistic                  -3.912981
p-value                          0.001941
#Lags Used                      13.000000
Number of Observations Used    118.000000
Critical Value (1%)             -3.487022
Critical Value (5%)             -2.886363
Critical Value (10%)            -2.580009
dtype: float64
"""
```

Trend를 제거하고 난 시계열에다가 1차 차분(1st order differencing)을 적용하여 Seasonality 효과를 다소 상쇄한 결과, p-value가 이전의 10% 정도까지로 줄어들었습니다. 데이터에 따라서는 2차 차분(2nd order differencing, 차분의 차분), 3차 차분(3rd order differencing, 2차 차분의 차분)을 적용하면 더욱 p-value를 낮출 수 있을지도 모른다.

### (2) 시계열 분해(Time Series Decomposition)

---

`statsmodels` 라이브러리 안에는 `seasonal_decompose` 메소드를 통해 시계열 안에 존재하는 trend, seasonality를 직접 분리해 낼 수 있는 기능이 있다. 이 기능을 활용하면 우리가 위에서 직접 수행했던 moving average 제거, differencing 등을 거치지 않고도 훨씬 안정적인 시계열을 분리해 낼 수 있게 된다.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.rcParams["figure.figsize"] = (11,6)
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
```

`Original` 시계열에서 `Trend`와 `Seasonality`를 제거하고 난 나머지를 `Residual`이라고 한다. 뒤집어서 말하면 `Trend+Seasonality+Residual=Original` 이 성립한다는 뜻이다. 이러한 Decomposing은 시계열 데이터를 이해하는 중요한 관점을 제시해 준다.

```python
# Residual 안정성 여부 확인
plt.rcParams["figure.figsize"] = (13,6)
plot_rolling_statistics(residual)

residual.dropna(inplace=True)
augmented_dickey_fuller_test(residual)

"""
Results of Dickey-Fuller Test:
Test Statistic                -6.332387e+00
p-value                        2.885059e-08
#Lags Used                     9.000000e+00
Number of Observations Used    1.220000e+02
Critical Value (1%)           -3.485122e+00
Critical Value (5%)           -2.885538e+00
Critical Value (10%)          -2.579569e+00
dtype: float64
"""
```

`Decomposing`을 통해 얻어진 `Residual`은 압도적으로 낮은 p-value를 보여 준다. 이 정도면 확실히 예측 가능한 수준의 안정적인 시계열이 얻어졌다고 볼 수 있을 것이다.

## ARIMA 모델의 개념

### (1) ARIMA 모델의 정의

---

이전 스텝에서 우리는 시계열 데이터가 Trend와 Seasonality와 Residual로 Decompose되며, Trend와 Seasonality를 효과적으로 분리해 낸 경우 아주 예측력 있는 안정적인 시계열 데이터로 변환 가능하다는 것을 확인하였다. 이런 원리를 이용하여 시계열 데이터 예측모델을 자동으로 만들어 주는 모델이 `ARIMA(Autoregressive Integrated Moving Average)`이다.

`ARIMA`는 `AR(Autoregressive)` + `I(Integrated)` + `MA(Moving Average)`가 합쳐진 모델이다.

**AR(자기회귀, Autoregressive)**

![images08.png](./images08.png)

- 자기회귀(AR)란, $Y_t$가 이전 p개의 데이터 $Y_{t-1},Y_{t-2}, ..., Y_{t-p}$의 가중합으로 수렴한다고 보는 모델이다.
- 가중치의 크기가 1보다 작은 $Y_{t-1},Y_{t-2}, ..., Y_{t-p}$의 가중합으로 수렴하는 자기회귀 모델과 안정적 시계열은 동계적으로 동치이다.
- AR은 일반적인 시계열에서 Trend와 Seasonality를 제거한 Residual에 해당하는 부분을 모델링한다고 볼 수 있다.
- 주식값이 항상 일정한 균형 수준을 유지할 것이라고 예측하는 관점이 바로 주식 시계열을 AR로 모델링하는 관점이라고 볼 수 있다.

**MA(이동평균, Moving Average)**

![images09.png](./images09.png)

- 이동평균(MV)은 $Y_t$가 이전 q개의 예측오차값 $e_{t-1},e_{t-2}, ..., e_{t-q}$의 가중합으로 수렴한다고 보는 모델이다.
- MA는 일반적인 시계열에서 Trend에 해당하는 부분을 모델링한다고 볼 수 있다. 예측오차값 $e_{t-1}$이 +라면 모델 예측보다 관측값이 더 높았다는 뜻이므로, 다음 $Y_t$ 예측 시에는 예측지를 올려잡게 된다.
- 주식값은 항상 최근의 증감 패턴이 지속될 것이라고 예측하는 관점이 바로 주식 시계열을 MA로 모델링하는 관점이라고 볼 수 있다.

**I (차분누적, Integration)**

- 차분누적은 $Y_t$ 이 이전 데이터와 d차 차분의 누적(integration) 합이라고 보는 모델이다.
- 예를 들어서 d=1이라면, $Y_t$ 는 $Y_{t-1}$과 $ΔY_{t-1}$의 합으로 보는 것이다.
- I는 일반적인 시계열에서 Seasonality에 해당하는 부분을 모델링한다고 볼 수 있다.

ARIMA는 위 3가지 모델을 모두 한꺼번에 고려하는 모델이다. 

### (2) ARIMA 모델의 모수 p, q, d

---

ARIMA를 활용해서 시계열 예측 모델을 성공적으로 만들기 위해서는 ARIMA의 모수(parameter)를 데이터에 맞게 설정해야 한다. 쉽게 말하자면 모델에 아주 핵심적인 숫자들을 잘 설정해야 올바른 예측식을 구할 수 있다는 것이다.

ARIMA의 모수는 3가지가 있는데, 자기회귀 모형(AR)의 시차를 의미하는 p, 차분(diffdrence) 횟수를 의미하는 d, 이동평균 모형(MA)의 시차를 의미하는 q가 있다.

이들 중 p 와 q 에 대해서는 통상적으로 p + q < 2, p * q = 0 인 값들을 사용하는데, 이는 p 나 q 중 하나의 값이 0이라는 뜻이다. 이렇게 하는 이유는 실제로 대부분의 시계열 데이터는 자기회귀 모형(AR)이나 이동평균 모형(MA) 중 하나의 경향만을 강하게 띠기 때문이다.

그러면 ARIMA(p,d,q) 모델의 모수를 결정하는 방법은 어떻게 될까? 예를 들어 q라면 이전 스텝에서 Moving Average를 구할 때의 window=12에 해당하는 값과 같은 역할을 한다는 느낌이 든다. 이 값을 어떻게 결정하느냐가 시계열 데이터의 안정성 및 이후 예측성능에 영향을 크게 미칠 것이다.

ARIMA의 적절한 모수 p,d,q를 선택하기 위한 방법에는 엄청난 통계학적인 다양한 시도들이 있다. 통계학적인 설명을 생략하고 결론부터 이야기하자면, 모수 p,d,q는 `ACF(Autocorrelation Function)`와 `PACF(Partial Autocorrelation Function)`을 통해 결정할 수 있다. 이 AutoCorrelation은 우리가 맨 첫 스텝에서 만났던 바로 개념 중 하나인 자기상관계수와 같은 것이다.

ACF 는 시차(lag)에 따른 관측치들 사이의 관련성을 측정하는 함수이며, PACF 는 다른 관측치의 영향력을 배제하고 두 시차의 관측치 간 관련성을 측정하는 함수이다.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(ts_log)   # ACF : Autocorrelation 그래프 그리기
plot_pacf(ts_log)  # PACF : Partial Autocorrelation 그래프 그리기
plt.show()

```

![images010.png](./images010.png)

![images011.png](./images011.png)

아래 그림은 `ACF`를 통해 MA 모델의 시차 q를 결정하고, `PACF`를 통해 AR 모델의 시차 p를 결정할 수 있음을 통계학적으로 설명하는 아티클에서 요약결론 부분만 가져온 것이다.

![images012.png](./images012.png)

이 결론에 따라 보자면 PACF 그래프를 볼 때 p=1이 매우 적합한 것 같다. p가 2 이상인 구간에서 PACF는 거의 0에 가까워지고 있기 때문이다. PACF가 0이라는 의미는 현재 데이터와 p 시점 떨어진 이전의 데이터는 상관도가 0, 즉 아무 상관 없는 데이터이기 때문에 고려할 필요가 없다는 뜻이다.

반면 ACF는 점차적으로 감소하고 있어서 AR(1) 모델에 유사한 형태를 보이고 있다. 

q에 대해서는 적합한 값이 없어 보인다. MA를 고려할 필요가 없다면 q=0으로 둘 수 있으나, q를 바꿔 가면서 확인해 보는 것도 좋을 것 같다.

d를 구하기 위해서는 좀 다른 접근이 필요하다. d차 차분을 구해 보고 이때 시계열이 안정된 상태인지를 확인해 보아야 한다.

```python
# 1차 차분 구하기
diff_1 = ts_log.diff(periods=1).iloc[1:]
diff_1.plot(title='Difference 1st')

augmented_dickey_fuller_test(diff_1)

"""
Results of Dickey-Fuller Test:
Test Statistic                  -2.717131
p-value                          0.071121
#Lags Used                      14.000000
Number of Observations Used    128.000000
Critical Value (1%)             -3.482501
Critical Value (5%)             -2.884398
Critical Value (10%)            -2.578960
dtype: float64
"""

# 2차 차분 구하기
diff_2 = diff_1.diff(periods=1).iloc[1:]
diff_2.plot(title='Difference 2nd')

augmented_dickey_fuller_test(diff_2)

"""
Results of Dickey-Fuller Test:
Test Statistic                -8.196629e+00
p-value                        7.419305e-13
#Lags Used                     1.300000e+01
Number of Observations Used    1.280000e+02
Critical Value (1%)           -3.482501e+00
Critical Value (5%)           -2.884398e+00
Critical Value (10%)          -2.578960e+00
dtype: float64
"""
```

이번 경우에는 1차 차분을 구했을 때 약간 애매한 수준의 안정화 상태를 보였고, 2차 차분을 구했을 때는 확실히 안정화 상태였지만 이번 경우에는 d=1로 먼저 시도해보자.

### (3) 학습데이터 분리

---

```python
# train, test 데이터 분리
train_data, test_data = ts_log[:int(len(ts_log)*0.9)], ts_log[int(len(ts_log)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.plot(ts_log, c='r', label='training dataset')  # train_data를 적용하면 그래프가 끊어져 보이므로 자연스러운 연출을 위해 ts_log를 선택
plt.plot(test_data, c='b', label='test dataset')
plt.legend()

# 데이터셋 형태 확인
print(ts_log[:2])
print(train_data.shape)
print(test_data.shape)
```

## ARIMA 모델 훈련과 추론

---

위에서 우리는 일단 p=1, d=1, q=0을 모수로 가지는 ARIMA 모델을 우선적으로 고려하게 되었다. ARIMA 모델을 훈련하는 것은 아래와 같이 간단하다.

```python
from statsmodels.tsa.arima_model import ARIMA

# Build Model
model = ARIMA(train_data, order=(1, 1, 0))  
fitted_m = model.fit(disp=-1)  
print(fitted_m.summary())

# 모델 예측 확인
fitted_m.plot_predict()

# Forecast : 결과가 fc에 담깁니다. 
fc, se, conf = fitted_m.forecast(len(test_data), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test_data.index)   # 예측결과
lower_series = pd.Series(conf[:, 0], index=test_data.index)  # 예측결과의 하한 바운드
upper_series = pd.Series(conf[:, 1], index=test_data.index)  # 예측결과의 상한 바운드

# Plot
plt.figure(figsize=(9,5), dpi=100)
plt.plot(train_data, label='training')
plt.plot(test_data, c='b', label='actual price')
plt.plot(fc_series, c='r',label='predicted price')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
plt.legend()
plt.show()
```

![images013.png](./images013.png)

최종적인 모델의 오차율을 계산하려면, 그동안 로그 변환된 시계열을 사용해 왔던 것을 모두 지수 변환하여 원본의 스케일로 계산해야 타당하다. np.exp()를 통해 전부 원본 스케일로 돌린 후 MSE, MAE, RMSE, MAPE를 계산한다.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

mse = mean_squared_error(np.exp(test_data), np.exp(fc))
print('MSE: ', mse)

mae = mean_absolute_error(np.exp(test_data), np.exp(fc))
print('MAE: ', mae)

rmse = math.sqrt(mean_squared_error(np.exp(test_data), np.exp(fc)))
print('RMSE: ', rmse)

mape = np.mean(np.abs(np.exp(fc) - np.exp(test_data))/np.abs(np.exp(test_data)))
print('MAPE: {:.2f}%'.format(mape*100))

"""
MSE:  5409.550103512347
MAE:  63.136923863759435
RMSE:  73.54964380275644
MAPE: 14.08%
"""
```

최종적으로 예측 모델의 메트릭으로 활용하기에 적당한 MAPE 기준으로 14% 정도의 오차율을 보였다. 만족스럽지 못한 결과인 것 같아서 더 적당한 모수를 찾아 개선하면 좋을 것 같다. q=8 을 줄 경우 MAPE가 10% 정도로 내려간다. q=12를 쓸 수 있으면 더욱 좋을 것 같지만 이번 경우에는 데이터셋이 너무 작아 쓸 수 없다.

## 주식 예측에 도전해보자

[야후 파이낸스](https://finance.yahoo.com/)에서 종목을 검색한 후, "Historical Data" 탭에서 "Time Period"를 "Max"로 선택, "Apply" 버튼을 눌러 과거 상장한 시점부터 가장 최근까지의 자료를 조회한 다음 "Download"를 클릭하면 데이터를 다운로드를 할 수 있다.

## 회고록

- 시계열 데이터를 처음 접했을 땐 다루기 어려운 데이터라고 생각했는데 노드를 진행하면서 계속 다루다보니 조금은 익숙해진 것 같다.
- 예전부터 주가를 예측하는 것은 한번 해보고 싶었는데 생각보다 어려운 것 같다. 하긴 쉬우면 누구나 부자가 됐겠지...
- 처음으로 raw data를 처리하는 과정에서 이상치를 발견하여 그동안 학습한 방법을 동원하여 정상적으로 수정하였다! 직접 해보고 나니 조금은 성장했다고 느꼈다.
- 이전에 해커톤을 진행했을 때도 시계열 데이터를 가지고 가격을 예측하는 문제에서 ARIMA 모델은 성능이 별로여서 XGBoost와 LightGBM을 이용했던 기억이 난다. 이 모델도 추후 앙상블 모델을 이용해서 한번 예측을 시도해봐야겠다.

### 유용한 링크

[http://www.dodomira.com/2016/04/21/arima_in_r/](http://www.dodomira.com/2016/04/21/arima_in_r/) ARIMA 모형

[https://destrudo.tistory.com/15](https://destrudo.tistory.com/15) 공분산과 상관계수

[https://rfriend.tistory.com/264](https://rfriend.tistory.com/264) 결측치 보간

[https://m.blog.naver.com/baedical/10109291879](https://m.blog.naver.com/baedical/10109291879) p-value
