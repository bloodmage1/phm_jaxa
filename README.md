# 경남 제조 해커톤 AI

## 1. 개발 목적

데이터 FFT 변환 및 AutoEncoder 방식을 통하여 대표 데이터를 생성하여 공정 부품 고장 예지 서비스, 이상 상황 탐지 서비스를 제공합니다. 해당 서비스를 통하여 공정 부품의 교체 시기, 이상 상황을 선제적으로 탐지함으로서, 예지보전 분야를 자동화하여 공정 효율을 향상시킬 수 있습니다.

## 2. 개발 목표와 내용

공정 부품 고장을 예지하고 예지한 고장 상황을 알립니다.현재 어떤 이상이 있는지 판단하는 분류 알고리즘, 언제 이상이 생길지 판단하는 AutoEncoder 모델을 이용하는 알고리즘, AutoEncoder 모델을 이용하여 계산한 재구성 손실값이 어떤 이상으로 진행되어가는지 판단하는 알고리즘입니다.

## 3. 순서도

1. 학습 – 충분한 양의 진동데이터를 가지고 AutoEncoder 모델의 학습(정상, 이물질투입진동, 균형이상)
2. 수집 – 예측하고자 하는 설비의 진동 데이터를 수집
3. 분류 – 분류 모델로 수집되는 데이터가 이상이 있는지 판단
4. 계산 – 정상 AutoEncoder 모델에 분류 판단이 끝난 Input data가 들어가 Output data가 만들어짐. Output data와 Input data 사이의 재구성 손실값을 계산
5. 예지 – 시간에 따른 재구성 손실값의 변화율을 이용하여 시설이 언제 고장이 날 것을 예지
6. 구분 – 4와 같은 방법으로 이물질투입 AutoEncoder 모델과 균형이상 AutoEncoder 모델에도 재구성손실값의 변화율을 계산. 그 값이 감소하는 모델에서 이상 유형 구분.


<img src="https://github.com/bloodmage1/PHM_AI/ppt/SUNSEODO.PNG" />


## 4. 사용 모델
[https://github.com/bloodmage1/PHM_AI/ppt/SUNSEODO.PNG] 제 AutoEncoder 모델을 참고해서 만들었습니다.

## 5. 전처리

FFT를 사용하여 모델에 이용할 수 있게끔 전처리 하였습니다. 사용한 코드는 다음과 같습니다.

```
def ae_with_fft(df, frequency):
    extracted_features = []
    for i in range(0, len(df), frequency):
        segment = df.iloc[i:i+frequency]
        if len(segment) == frequency:
            # FFT 적용
            fft_values = np.fft.fft(segment)
            
            half_length = len(fft_values) // 2
            fft_values_half = fft_values[:half_length]
            
            magnitude = np.abs(fft_values_half)
            extracted_features.append(magnitude)
    return pd.DataFrame(extracted_features)
```

## 6. 결과

<img src="https://github.com/bloodmage1/PHM_AI/ppt/result_.PNG" />

그림을 통해 정상파일과 비교하여 error 파일이 다음과 같이 차이가 나는 것을 알 수 있습니다.
