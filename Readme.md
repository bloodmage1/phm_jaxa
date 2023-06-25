# 압력 데이터를 이용한 PHM

# 목차

1. MFCC
   - MFCC 전처리 과정
   - 샘플링과 윈도윙

2. FFT
   - 데이터에 포함된 비언어적 표현

# MFCC

**Mel-Frequency Cepstral Coefficient**의 약자로 한글로 풀이하면 멜 주파수 중심 계수이다. Mel은 사람의 달팽이관을 모티브로 따온 값이라고 생각하면 된다. 달팽이관의 각 부분은 각기 다른 진동수를 감지한다. 이 달팽이관이 감지하는 진동수를 기반으로 하여 사람은 소리를 인식한다.

주파수를 특징으로 쓰는 이유다. 하지만 이것에도 특징이 있는데 주파수가 낮은 대역에서는 주파수의 변화를 잘 감지하는데, 주파수가 높은 대역에서는 주파수의 변화를 잘 감지하지 못한다. 달팽이관에서는 저주파 대역을 감지하는 부분은 굵지만 고주파 대역을 감지하는 부분으로 갈수록 얇아지기 때문이다. 위와 같은 <span style="color:skyblue">**달팽이관 특성을 고려한 값을 Mel-scale**</span>이라고 한다.

데이터에서 주파수를 성분으로 뽑아내야 한다면 푸리에변환을 해야 할 것이다. 하지만 사람이 발성하는 음성은 그 길이가 천차만별일 것이다. 안녕하세요라고 말하더라도, 어떤 사람은 0.5초, 어떤 사람은 1초가 걸릴 수도 있다. 그래서 음성 데이터에서 한 번에 멜 스케일을 뽑게 되면 이 천차만별인 길이에 대하여 같은 안녕하세요라는 음성이라고 학습시키기는 어려울 것이다.

위와 같은 문제를 해결하기 위해 음성데이터를 모두 20에서 40ms로 쪼갠다. 여기서 사람의 음성은 20에서 40ms 사이에서는 음소가 바뀔 수 없다는 연구결과들을 기반으로 음소는 해당시간 내에 바뀔 수 없다고 가정한다. 그래서 <span style="color:skyblue">**MFCC는 음성데이터를 모두 20~40MS 단위로 쪼개고 쪼갠 단위에 대해서 Mel 값을 뽑아서 특징으로 사용**<span>한다.

## MFCC 전처리 과정

![Example instance](https://github.com/bloodmage1/Multimodal_paper/blob/main/mfcc_preprocessing.png)

사람이 발성 시 몸의 구조 때문에 실제로 낸 소리에서 고주파 성분은 많이 줄어들게 되서 나온다고 한다. 그래서 <span style="color:skyblue">**먼저 줄어든 고주파 성분을 변조가 강하게 걸리도록 High-pass Filter를 적용해주는 과정이 포함**</span>된다.

## 샘플링과 윈도윙

Pre-emphasis된 신호에 대해서 앞에서 언급했던 이유 때문에 신호를 20에서 40ms 단위의 프레임으로 분할한다. 여기서 프레임을 50% 겹치게 분할한다(프레임끼리 연속성을 만들어주기 위해).

만약 프레임이 서로 뚝뚝 떨어지게 샘플링을 한다면, 프레임과 프레임의 접합 부분에서 순간 변화율이 무한대가 될 수 있기 때문이다.

그리고 이 프레임들에 대해 window를 각각 적용한다. 이때 <span style="color:skyblue">**[hamming window](https://velog.io/@workhard/hanning-window%EB%9E%80)를 적용**</span>한다.

# FFT(핵심)

하지만 Mel Scale을 사용하지 않아도 될 상황이 존재한다.

<span style="color:skyblue">**주파수 정보가 중요하지 않은 경우**</span>: 분석하려는 신호가 주로 음악이나 음성이 아닌 다른 유형의 신호인 경우, 예를 들어 기계 소음이나 환경 소리 등의 경우에는 Mel 스케일을 사용하는 것보다 FFT만을 사용하는 것이 더 적합할 수 있다.

<span style="color:skyblue">**특정 주파수 대역의 정보만 필요한 경우**</span>: 분석하려는 신호에서 특정 주파수 대역의 정보만 필요한 경우에는 Mel 스케일을 사용하지 않아도 될 수 있다. 예를 들어, 주파수 대역 중에서 특정 주파수 범위의 세부 정보만을 관심있게 분석해야 하는 경우에는 FFT만을 사용하여 해당 주파수 대역을 추출할 수 있다.

<span style="color:skyblue">**추가적인 데이터 처리가 필요한 경우**</span>: Mel 스케일을 적용하는 것은 FFT 결과를 조금 더 인간의 청각에 적합한 척도로 변환하는 것이다. 따라서 Mel 스케일을 적용하는 것은 데이터 처리에 추가적인 단계를 필요로 한다. 때로는 이러한 추가 단계가 필요하지 않거나 비용이 많이 들 때, FFT만을 사용하여 간단하게 분석할 수 있는 경우도 있다.

## 데이터에 포함된 비언어적 표현

[데이터](https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR)에 포함된 비언어적 표현은 다음과 같다.

- c/ : 휴지구간이 확보되지 않은 연속발성(0.3초 미만) 

- n/ : 발성 이외의 단발적인 소음 

- N/ : 음성 구간의 50% 이상 잡음이 포함된 상황 

- u/ : 단어의 내용을 알아 들을 수 없는 상황

-  l/ : 발성중 음음 소리가 포함된 상황 (small 'L')

- b/ : 발성 중 숨소리, 김침 소리가 포함된 상황

- * : 단어 중 일부만 알아 듣거나 알아들었으나 애매한 상황

- + : 발성 중 말을 반복적으로 더듬는 상황

- / : 간투사