---
title: "[논문리뷰] NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers"
last_modified_at: 2023-08-12
categories:
  - 논문리뷰
tags:
  - Diffusion
  - Text-to-Speech
  - Voice Conversion
  - Audio and Speech Processing
  - AI
  - Microsoft
excerpt: "NaturalSpeech 2 논문 리뷰"
use_math: true
classes: wide
---

> arXiv 2021. [[Paper](https://arxiv.org/abs/2304.09116)] [[Page](https://speechresearch.github.io/naturalspeech2/)]  
> Kai Shen, Zeqian Ju, Xu Tan, Yanqing Liu, Yichong Leng, Lei He, Tao Qin, Sheng Zhao, Jiang Bian  
> Microsoft Research Asia & Microsoft Azure Speech  
> 18 Apr 2023  

## Introduction
인간의 말은 다양한 speaker identity (ex. 성별, 억양, 음색), 운율, 스타일 (ex.: 말하기, 노래) 등으로 다양하다. TTS는 자연스럽고 사람과 유사한 음성을 좋은 품질과 다양성으로 합성하는 것을 목표로 한다. 신경망과 딥러닝의 발달로 TTS 시스템은 명료도와 자연스러움 측면에서 우수한 음성 품질을 달성했으며 일부 시스템 (ex. [NaturalSpeech](https://kimjy99.github.io/논문리뷰/naturalspeech))은 단일 speaker 녹음 스튜디오 벤치마킹 데이터셋 (ex. LJSpeech)에서 인간 수준의 음성 품질을 달성하기도 한다. 전체 TTS 커뮤니티가 음성 명료도와 자연스러움 측면에서 큰 성과를 거둔 것을 감안할 때 이제 자연스럽고 사람과 같은 음성을 합성하기 위해 음성 다양성이 점점 더 중요해지는 TTS의 새로운 시대에 접어들었다.

이전의 spekaer 제한 녹음 스튜디오 데이터셋은 제한된 데이터 다양성으로 인해 인간 음성의 다양한 speaker identity, 운율, 스타일을 캡처하기에 충분하지 않다. 대신 대규모 코퍼스에서 TTS 모델을 학습하여 이러한 다양성을 학습할 수 있으며, 부산물로 이러한 학습된 모델은 few-shot 또는 zero-shot 기술을 사용하여 무제한의 처음 보는 시나리오로 일반화할 수 있다. 현재 대규모 TTS 시스템은 일반적으로 연속 음성 파형을 이산 토큰으로 양자화하고 autoregressive 언어 모델로 이러한 토큰을 모델링한다. 이 파이프라인에는 몇 가지 제한 사항이 있다. 

1. 음성 (이산 토큰) 시퀀스는 일반적으로 매우 길고 (10초 음성에는 일반적으로 수천 개의 토큰이 있음) autoregressive 모델은 오차 전파로 인해 불안정한 음성 출력이 발생한다. 
2. 코덱과 언어 모델 사이에 딜레마가 있다. 한편으로는 토큰 양자화를 사용하는 코덱 (VQ-VAE 또는 VQ-GAN)은 일반적으로 낮은 비트레이트 토큰 시퀀스를 가지며, 이는 언어 모델 생성을 용이하게 하지만 고주파수 미세 음향 디테일에 대한 정보 손실을 초래한다. 다른 한편으로는 일부 개선 방법은 음성 프레임을 표현하기 위해 여러 개의 residual discrete 토큰을 사용하는데, 이는 flatten될 경우 토큰 시퀀스의 길이를 여러 번 증가시키고 언어 모델링에 어려움을 초래한다.

본 논문에서는 표현적인 운율, 우수한 robustness, 그리고 가장 중요한 음성 합성을 위한 강력한 zero-shot 능력을 달성하기 위해 [latent diffusion model (LDM)](https://kimjy99.github.io/논문리뷰/ldm)을 갖춘 TTS 시스템인 **NaturalSpeech 2**를 제안한다. 먼저 음성 파형을 코덱 인코더를 사용하여 일련의 latent 벡터로 변환하는 뉴럴 오디오 코덱을 학습하고 코덱 디코더를 사용하여 이러한 latent 벡터에서 음성 파형을 재구성한다. 오디오 코덱을 학습한 후 코덱 인코더를 사용하여 학습 세트의 음성에서 latent 벡터를 추출하고 음소 인코더, duration predictor, pitch predictor에서 얻은 사전 벡터으로 컨디셔닝하는 LDM의 타겟으로 사용한다. Inference하는 동안 먼저 LDM을 사용하여 텍스트/음소 시퀀스에서 latent 벡터를 생성한 다음 코덱 디코더를 사용하여 이러한 latent 벡터에서 음성 파형을 생성한다.

<center><img src='{{"/assets/img/naturalspeech2/naturalspeech2-table1.PNG" | relative_url}}' width="80%"></center>
<br>
NaturalSpeech 2의 일부 디자인 선택은 다음과 같다. (위 표 참조)

- **이산 토큰 대신 연속 벡터**: 뉴럴 코덱의 음성 재구성 품질을 보장하기 위해 이전 연구들은 일반적으로 여러 residual quantizer로 음성을 양자화하였다. 결과적으로, 획득된 이산 토큰 시퀀스는 매우 길고 (예를 들어, 각 음성 프레임에 대해 8개의 residual quantizer를 사용하는 경우 결과로 생성되는 flatten된 토큰 시퀀스는 8배 더 길어짐) 음향 모델 (autoregressive 언어 모델)에 많은 압력을 가한다. 따라서 이산 토큰 대신 연속 벡터를 사용하여 시퀀스 길이를 줄이고 세분화된 음성 재구성을 위한 정보량을 늘릴 수 있다.
- **Autoregressive model 대신 diffusion model**: Diffusion model을 활용하여 non-autoregressive 방식으로 연속 벡터의 복잡한 분포를 학습하고 autoregressive model에서의 오차 전파를 방지한다.
- **In-context learning을 위한 음성 프롬프팅 메커니즘**: Diffusion model이 음성 프롬프트의 특성을 따르고 zero-shot 능력을 향상시키기 위해 diffusion model과 pitch/duration predictor에서 in-context learning을 용이하게 하는 음성 프롬프팅 메커니즘을 설계하였다.

이러한 설계의 이점을 활용하여 NaturalSpeech 2는 이전의 autoregressive model보다 더 안정적이고 robust하며, 2단계 토큰 예측 대신 하나의 음향 모델(diffusion model)만 필요하며 duration/pitch 예측과 non-autoregressive 생성으로 인해 음성을 넘어 (ex. 노래하는 음성) 스타일을 확장할 수 있다.

## NaturalSpeech 2
<center><img src='{{"/assets/img/naturalspeech2/naturalspeech2-fig1.PNG" | relative_url}}' width="75%"></center>
<br>
위 그림에서 볼 수 있듯이 NaturalSpeech 2는 뉴럴 오디오 코덱 (인코더, 디코더)과 prior (음소 인코더, duration/pitch predictor)가 있는 diffusion model로 구성된다. 음성 파형은 복잡하고 고차원이기 때문에 재생성 학습의 패러다임에 따라 먼저 오디오 코덱 인코더를 사용하여 음성 파형을 latent 벡터로 변환하고 오디오 코덱 디코더를 사용하여 latent 벡터에서 음성 파형을 재구성한다. 다음으로 diffusion model을 사용하여 텍스트/음소 입력으로 컨디셔닝된 latent 벡터를 예측한다.

### 1. Neural Audio Codec with Continuous Vectors
연속 벡터가 포함된 오디오 코덱은 다음과 같은 몇 가지 이점이 있다. 

1. 연속 벡터는 이산 토큰보다 압축률이 낮고 비트 전송률이 높아 고품질 오디오 재구성을 보장할 수 있다. 
2. Hidden 시퀀스의 길이를 늘리지 않는 이산 양자화에서와 같이 각 오디오 프레임에는 여러 토큰 대신 하나의 벡터만 있다.

<center><img src='{{"/assets/img/naturalspeech2/naturalspeech2-fig2.PNG" | relative_url}}' width="95%"></center>
<br>
위 그림에서 볼 수 있듯이 뉴럴 오디오 코덱은 오디오 인코더, residual vector-quantizer (RVQ), 오디오 디코더로 구성된다. 

1. 오디오 인코더는 16KHz 오디오에 대해 총 다운샘플링 속도가 200인 여러 convolution 블록으로 구성된다. 즉, 각 프레임은 12.5ms 음성 세그먼트에 해당한다. 
2. RVQ는 오디오 인코더의 출력을 [SoundStream](https://arxiv.org/abs/2107.03312)을 따라 여러 residual vector로 변환한다. 이러한 residual vector의 합은 diffusion model의 학습 타겟으로 사용되는 양자화된 벡터로 간주된다. 
3. 오디오 디코더는 양자화된 벡터에서 오디오 파형을 생성하는 오디오 인코더의 구조를 반영한다. 

뉴럴 오디오 코덱의 흐름은 다음과 같다.

$$
\begin{aligned}
&\textrm{Audio Encoder}: h = f_\textrm{enc} (x) \\
&\textrm{RVQ}: \{e_j^i\}_{j=1}^R = f_\textrm{rvq} (h^i), \; z^i = \sum_{j=1}^R e_j^i, \; z = \{z^i\}_{i=1}^n \\
&\textrm{Audio Decoder}: x = f_\textrm{dec} (z)
\end{aligned}
$$

여기서 $f_\textrm{enc}$, $f_\textrm{rvq}$, $f_\textrm{dec}$는 오디오 인코더, RVQ, 오디오 디코더를 나타냔다. $x$는 음성 파형이고, $h$는 프레임 길이가 $n$인 오디오 인코더에서 얻은 hidden 시퀀스이고, $z$는 $h$와 동일한 길이를 가진 양자화된 벡터 시퀀스이다. $i$는 음성 프레임의 인덱스, $j$는 residual quantizer의 인덱스, $R$은 residual quantizer의 총 개수, $e_j^i$는 $i$번째 hidden 프레임 $h^i$에서 $j$번째 residual quantizer에 의해 획득된 코드북 ID의 임베딩 벡터이다. 뉴럴 코덱의 학습은 [SoundStream](https://arxiv.org/abs/2107.03312)의 loss function을 따른다.

실제로 연속 벡터를 얻으려면 벡터 양자화가 필요하지 않고 오토인코더 또는 VAE만 필요하다. 그러나 정규화와 효율성을 위해 매우 많은 수의 quantizer와 코드북 토큰이 있는 RVQ를 사용하여 연속 벡터를 근사화한다. 이렇게 하면 두 가지 이점이 있다. 

1. [LDM](https://kimjy99.github.io/논문리뷰/ldm)을 학습할 때 메모리 비용인 연속 벡터를 저장할 필요가 없다. 연속 벡터를 도출하는 데 사용되는 코드북 임베딩과 양자화 토큰 ID만 저장한다. 
2. 연속 벡터를 예측할 때 이러한 양자화 토큰 ID를 기반으로 이산 분류에 regularization loss을 추가할 수 있다.

### 2. Latent Diffusion Model with Non-Autoregressive Generation
Diffusion model을 활용하여 텍스트 시퀀스 $y$로 컨디셔닝된 양자화된 latent 벡터 $z$를 예측한다. 음소 인코더, duration predictor, pitch predictor로 구성된 이전 모델을 활용하여 텍스트 입력을 처리하고 diffusion model의 조건으로 보다 유익한 hidden 벡터 $c$를 제공한다.

#### Diffusion Formulation
Diffusion (forward) process와 denoising (reverse) process를 각각 확률적 미분 방정식(SDE)으로 공식화한다. Forward SDE는 뉴럴 코덱에서 얻은 latent 벡터 $z_0$를 Gaussian noise로 변환한다.

$$
\begin{equation}
dz_t = -\frac{1}{2} \beta_t z_t + \sqrt{\beta_t} dw_t, \quad t \in [0,1]
\end{equation}
$$

여기서 $w_t$는 표준 브라운 운동이고 $\beta_t$는 음이 아닌 noise schedule 함수이다. 그러면 솔루션은 다음과 같다.

$$
\begin{equation}
z_t = \exp (- \frac{1}{2} \int_0^t \beta_s ds) z_0 + \int_0^t \sqrt{\beta_s} \exp (-\frac{1}{2} \int_0^t \beta_u du) dw_s
\end{equation}
$$

Ito 적분의 속성에 의해 $z_0$가 주어졌을 때 $z_t$의 조건부 분포은 가우시안이다. 

$$
\begin{equation}
p(z_t \vert z_0) \sim \mathcal{N}(\rho (z_0, t), \Sigma_t) \\
\textrm{where} \quad \rho (z_0, t) = \exp (- \frac{1}{2} \int_0^t \beta_s ds) z_0, \quad \Sigma_t = I - \exp (- \int_0^t \beta_s ds)
\end{equation}
$$

Reverse SDE는 다음과 같은 프로세스를 통해 Gaussian noise를 데이터 $z_0$로 다시 변환한다.

$$
\begin{equation}
dz_t = - (\frac{1}{2} z_t + \nabla p_t (z_t)) \beta_t dt + \sqrt{\beta_t} d \tilde{w}_t, \quad t \in [0, 1]
\end{equation}
$$

여기서 $\tilde{w}$는 역시간 브라운 운동이다. 또한 reverse process에서 상미분 방정식(ODE)을 고려할 수 있다.

$$
\begin{equation}
dz_t = - (\frac{1}{2} z_t + \nabla p_t (z_t)) \beta_t dt, \quad t \in [0, 1]
\end{equation}
$$

신경망 $s_\theta$를 학습시켜 score $\nabla \log p_t (z_t)$를 추정한 다음 Gaussian noise $z_1 \sim \mathcal{N} (0, 1)$에서 시작하여 SDE 또는 ODE를 수치적으로 풀어 데이터 $z_0$를 샘플링할 수 있다. 신경망 $s_\theta (z_t, t, c)$는 현재 noisy 벡터 $z_t$, timestep $t$, 조건 정보 $c$를 취하는 WaveNet을 기반으로 한다. $s_\theta$는 score 대신 데이터 $$\hat{z}_0$$를 예측하여 음성 품질이 더 나은 결과를 얻었다. 따라서 $$\hat{z}_0 = s_\theta (z_t, t, c)$$이다. Diffusion model 학습을 위한 loss function은 다음과 같다.

$$
\begin{aligned}
\mathcal{L}_\textrm{diff} = \mathbb{E}_{z_0, t} [ & \|\hat{z}_0 - z_0 \|_2^2 \\
+ \;& \| \Sigma_t^{-1} (\rho (\hat{z}_0, t) - z_t) - \nabla \log p_t (z_t) \|_2^2 \\
+ \;& \lambda_\textrm{ce-rvq} \mathcal{L}_\textrm{ce-rvq}]
\end{aligned}
$$

여기서 첫 번째 항은 데이터 loss이다. 두 번째 항은 score loss이며 예측 score는 $$\Sigma_t^{-1} (\rho(\hat{z}_0, t) − z_t)$$로 계산되며 inference에서 역 샘플링에도 사용된다. 세 번째 항 $$\mathcal{L}_\textrm{ce-rvq}$$는 RVQ에 기반한 새로운 cross-entropy (CE) loss이다. 특히, 각 residual quantizer $j \in [1, R]$에 대해 먼저 residual vector $$\hat{z}_0 − \sum_{i=1}^{j-1} e_i$$를 얻는다. 여기서 $e_i$는 $i$번째 residual quantizer의 ground-truth 양자화된 임베딩이다. 그런 다음 각 코드북이 quantizer $j$에 임베딩된 residual vector 사이의 L2 거리를 계산하고 softmax 함수로 확률 분포를 구한 다음 ground-truth 양자화된 임베딩의 ID $e_j$와 이 확률 분포 사이의 cross-entropy loss를 계산한다. $$\mathcal{L}_\textrm{ce-rvq}$$는 모든 $R$개의 residual quantizer에서 cross-entropy loss의 평균이고 $\lambda_\textrm{ce-rvq}$는 학습 중에 0.1로 설정된다.

#### Prior Model: Phoneme Encoder and Duration/Pitch Predictor
음소 인코더는 표준 feed-forward network가 convolution network로 수정되어 음소 시퀀스의 로컬 의존성을 캡처하는 여러 Transformer 블록으로 구성된다. Duration과 pitch 예측 변수는 모두 여러 convolution 블록과 동일한 모델 구조를 공유하지만 모델 파라미터는 다르다. L1 duration loss $$\mathcal{L}_\textrm{dur}$$와 pitch loss $$\mathcal{L}_\textrm{pitch}$$와 함께 duration과 pitch 예측 변수를 학습하기 위한 목적 함수로 ground-truth duration과 pitch 정보가 사용된다. 학습 시 ground-truth duration을 사용하여 음소 인코더에서 hidden 시퀀스를 확장하여 프레임 레벨의 hidden 시퀀스를 얻은 다음 프레임 레벨의 hidden 시퀀스에 ground-truth pitch 정보를 추가하여 최종 조건 정보 $c$를 얻는다. Inference하는 동안 해당 duration과 pitch가 사용된다.

Diffusion model의 총 loss function은 다음과 같다.

$$
\begin{equation}
\mathcal{L} = \mathcal{L}_\textrm{diff} + \mathcal{L}_\textrm{dur} + \mathcal{L}_\textrm{pitch}
\end{equation}
$$

### 3. Speech Prompting for In-Context Learning
더 나은 zero-shot 생성을 위한 in-context learning을 용이하게 하기 위해 음성 프롬프트의 다양한 정보(ex. speaker ID)를 따르도록 duration/pitch predictor와 diffusion model을 장려하는 음성 프롬프팅 메커니즘을 설계하였다. 음성 latent 시퀀스 $z$의 경우 음성 프롬프트로 프레임 인덱스가 $u$에서 $v$인 세그먼트 $z^{u:v}$를 무작위로 잘라내고 나머지 음성 세그먼트 $z^{1:u}$와 $z^{v:n}$을 concat하여 diffusion model의 학습 타겟으로 새 시퀀스 $z^{\ u:v}%$를 형성한다. 

<center><img src='{{"/assets/img/naturalspeech2/naturalspeech2-fig3.PNG" | relative_url}}' width="80%"></center>
<br>
위 그림과 같이 Transformer 기반 프롬프트 인코더를 사용하여 음성 프롬프트 $z^{u:v}$ (그림에서 $z^p$)를 처리하여 hidden 시퀀스를 얻는다. 이 hidden 시퀀스를 프롬프트로 활용하기 위해 duration/pitch 예측 변수와 diffusion model에 대한 두 가지 다른 전략이 있다. 

1. Duration과 pitch 예측 변수의 경우 query는 convolution layer의 hidden 시퀀스이고 key와 value는 프롬프트 인코더의 hidden 시퀀스인 convolution layer에 Q-K-V attention layer를 삽입한다. 
2. Diffusion model의 경우, diffusion model에 너무 많은 디테일을 노출하고 생성에 해를 끼칠 수 있는 프롬프트 인코더의 hidden 시퀀스에 직접 attend하는 대신 두 개의 attention block을 설계한다. 첫 번째 attention block에서 임의로 초기화된 $m$개의 임베딩을 query 시퀀스로 사용하여 프롬프트 hidden 시퀀스에 attend하고 attention 결과로 길이 $m$의 hidden 시퀀스를 얻는다. 두 번째 attention block에서는 WaveNet 레이어의 hidden 시퀀스를 query로 활용하고 길이 $m$의 attention 결과를 key와 value로 활용한다. 두 번째 attention block의 attention 결과를 FiLM 레이어의 조건부 정보로 사용하여 diffusion model에서 WaveNet의 hidden 시퀀스에 대한 affine transform을 수행한다. 

### 4. Connection to NaturalSpeech
NaturalSpeech 2는 [NaturalSpeech](https://kimjy99.github.io/논문리뷰/naturalspeech) 시리즈의 고급 버전이다. 이전 버전인 NaturalSpeech와 비교하여 NaturalSpeech 2는 다음과 같은 연결점과 차이점이 있다. 

1. 목표: NaturalSpeech 1과 2 모두 자연스러운 음성 합성을 목표로 하지만 초점이 다르다. NaturalSpeech는 사람이 녹음한 것과 동등한 음성을 합성하고 단일 speaker 녹음 스튜디오 데이터셋 (ex. LJSpeech)만 처리하여 음성 품질에 중점을 둔다. NaturalSpeech 2는 대규모, multi-speaker, wild 데이터셋를 기반으로 zero-shot 합성 능력을 탐색하여 음성 다양성에 중점을 둔다. 
2. 아키텍처: NaturalSpeech 2는 파형 재구성을 위한 인코더, 디코더와 이전 모듈 (음소 인코더, duration/pitch predictor)과 같은 NaturalSpeech의 기본 구성 요소를 유지한다. 그러나 대규모 음성 데이터셋에서 복잡하고 다양한 데이터 분포를 캡처하기 위해 모델링 능력을 높이기 위한 diffusion model, 재구성 품질과 예측 난이도를 절충하기 위해 latent 벡터를 정규화하는 RVQ, zero-shot 능력을 가능하게 하는 음성 프롬프팅 메커니즘을 활용한다.

## Experiments
- 데이터셋: Multilingual LibriSpeech (MLS) (16kHz)
- 모델 구성
  - 음소 인코더: 6-layer Transformer
    - attention head 개수: 8
    - 임베딩 차원: 512
    - 1D convolution filter size: 2048
    - convolution 1D kernel size: 9
    - dropout: 0.1
  - pitch/duration predictor
    - 30-layer 1D convolution (ReLU, layer normalization)
    - Q-K-V attention layer 10개 (attention head 8개, 512 hidden 차원)
    - 1D convolution layer 3개마다 attention layer 1개
    - dropout: 0.5
  - 음성 프롬프트 인코더: 6-layer Transformer (음소 인코더와 동일)
  - 프롬프팅 메커니즘: 토큰 수 $m$ = 32, hidden 차원 = 512
  - diffusion model: WaveNet 레이어 40개
    - 1D dilated convolution layer (kernel size = 3, filter size = 1024, dilation size = 2)
    - WaveNet 레이어 3개마다 FiLM layer 배치
    - hidden size: 512
    - dropout: 0.2
- 학습
  - 오디오 코덱
    - NVIDIA TESLA V100 16GB GPU 8개
    - batch size: GPU당 오디오 200개 (44만 step)
    - optimizer: Adam
    - learning rate: $2 \times 10^{-4}$
  - diffusion model
    - NVIDIA TESLA V100 32GB GPU 16개
    - batch size: GPU당 latent 벡터의 6천 프레임 (30만 step)
    - optimizer: AdamW
    - learning rate: $5 \times 10^{-4}$ (warmup 3.2만 step, inverse square root learning schedule)
- Inference
  - temperature $\tau = 1.2^2$를 사용하여 $z_T$를 $\mathcal{N} (0, \tau^{-1} I)$에서 샘플링
  - 생성 품질과 생성 속도 사이의 균형을 위해 Euler ODE solver를 채택 (diffusion step = 150)

### 1. Generation Quality
다음은 LibriSpeech와 VCTK에 대한 CMOS 결과이다.

<center><img src='{{"/assets/img/naturalspeech2/naturalspeech2-table3.PNG" | relative_url}}' width="36%"></center>

### 2. Generation Similarity
다음은 pitch와 duration의 평균(Mean), 표준 편차(Std), 왜도(Skew), 첨도(Kurt) 차이 측면에서 합성 음성과 프롬프트 음성 간의 운율 유사성을 비교한 표이다. 

<center><img src='{{"/assets/img/naturalspeech2/naturalspeech2-table4.PNG" | relative_url}}' width="70%"></center>
<br>
다음은 LibriSpeech와 VCTK에서의 SMOS이다.

<center><img src='{{"/assets/img/naturalspeech2/naturalspeech2-table5.PNG" | relative_url}}' width="36%"></center>

### 3. Robustness
다음은 LibriSpeech와 VCTK에서의 단어 오차율 (WER)이다.

<center><img src='{{"/assets/img/naturalspeech2/naturalspeech2-table6.PNG" | relative_url}}' width="36%"></center>
<br>
다음은 50개의 특정 어려운 문장에서 다른 autoregressive (AR) / non-autoregressive (NAR) 모델들과 비교한 표이다.

<center><img src='{{"/assets/img/naturalspeech2/naturalspeech2-table7.PNG" | relative_url}}' width="70%"></center>

### 4. Comparison with Other TTS Systems
다음은 SMOS와 CMOS를 VALL-E와 비교한 표이다.

<center><img src='{{"/assets/img/naturalspeech2/naturalspeech2-table8.PNG" | relative_url}}' width="31%"></center>

### 5. Ablation Study
다음은 ablation study 결과이다.

<center><img src='{{"/assets/img/naturalspeech2/naturalspeech2-table9.PNG" | relative_url}}' width="80%"></center>
<br>
다음은 pitch와 duration의 평균(Mean), 표준 편차(Std), 왜도(Skew), 첨도(Kurt) 차이 측면에서 다양한 길이의 합성 음성과 프롬프트 음성 간의 NaturalSpeech 2 운율 유사성을 비교한 표이다. 

<center><img src='{{"/assets/img/naturalspeech2/naturalspeech2-table10.PNG" | relative_url}}' width="69%"></center>

### 6. Zero-Shot Singing Synthesis
저자들은 노래 데이터 수집을 위해 웹에서 여러 노래하는 목소리와 짝을 이룬 가사를 크롤링하였다. 노래 데이터 전처리를 위해 음성 처리 모델을 활용하여 노래에서 반주를 제거하고 ASR 모델을 사용하여 정렬이 어긋난 샘플을 필터링한다. 그런 다음 데이터셋은 음성 데이터와 동일한 프로세스를 사용하여 구성되며 궁극적으로 약 30시간의 노래 데이터를 포함한다. 데이터셋은 업샘플링되고 음성 데이터와 혼합된다.

음성과 노래 데이터를 함께 사용하여 $5 \times 10^{-5}$의 learning rate로 NaturalSpeech 2를 학습시킨다. 더 나은 결과를 위해 diffusion step을 1000으로 설정한다. 노래하는 목소리를 합성하기 위해 노래를 부르는 다른 목소리의 ground-truth pitch와 duration를 사용하고 다른 노래 프롬프트를 사용하여 다른 가수 음색으로 노래하는 목소리를 생성한다. NaturalSpeech 2는 음성을 프롬프트로 사용하여 새로운 노래 음성을 생성할 수 있다.

### 7. Extension to Voice Conversion
Zero-shot TTS와 노래 합성 외에도 NaturalSpeech 2는 프롬프트 오디오 $z_\textrm{prompt}$의 음성을 사용하여 소스 오디오 $z_\textrm{source}$를 타겟 오디오 $z_\textrm{target}$으로 변환하는 것을 목표로 하는 zero-shot 음성 변환도 지원한다. 먼저 source-aware diffusion process를 사용하여 소스 오디오 $z_\textrm{source}$를 유익한 Gaussian noise $z_1$로 변환하고 다음과 같이 target-aware denoising process를 사용하여 타겟 오디오 $z_\textrm{target}$을 생성한다.

#### Source-Aware Diffusion Process
음성 변환에서 생성 프로세스를 용이하게 하기 위해 타겟 오디오에 대한 소스 오디오에서 필요한 정보를 제공하는 것이 도움이 된다. 따라서 일부 Gaussian noise로 소스 오디오를 직접 diffuse시키는 대신 소스 오디오의 일부 정보를 여전히 유지하는 시작점으로 소스 오디오를 diffuse한다. 특히 Diffusion Autoencoder의 확률적 인코딩 프로세스에서 영감을 받아 다음과 같이 $z_\textrm{source}$에서 시작점 $z_1$을 얻는다.

$$
\begin{equation}
z_1 = z_0 + \int_0^1 - \frac{1}{2} (z_t + \Sigma_t^{-1} (\rho (\hat{s}_\theta (z_t, t, c), t) - z_t)) \beta_t dt
\end{equation}
$$

여기서 $\Sigma_t^{-1} (\rho (\hat{s}_\theta (z_t, t, c), t) - z_t)$는 $t$에서의 예측 score이다. 이 프로세스는 denoising process에서 ODE의 역으로 생각할 수 있다.

#### Target-Aware Denoising Process
임의의 Gaussian noise에서 시작하는 TTS와 달리 음성 변환의 denoising process는 source-aware diffusion process에서 얻은 $z_1$부터 시작한다. TTS에서와 같이 표준 denoising process를 실행하여 $c$와 프롬프트 오디오 $z_\textrm{prompt}$로 컨디셔닝된 최종 타겟 오디오 $z_\textrm{target}$을 얻는다. 여기서 $c$는 소스 오디오의 음소 및 duration 시퀀스와 예측된 pitch 시퀀스에서 가져온다.

결과적으로 NaturalSpeech 2는 소스 음성과 유사한 운율을 나타내는 음성을 생성하는 동시에 프롬프트에서 지정한 음색을 복제할 수 있다.

### 8. Extension to Speech Enhancement
NaturalSpeech 2는 음성 변환의 확장과 유사한 음성 향상으로 확장될 수 있다. 이 세팅에서는 배경 소음을 포함하는 소스 오디오 $$z'_\textrm{source}$$, source-aware diffusion process를 위한 배경 소음이 있는 프롬프트 $$z'_\textrm{prompt}$$, target-aware denoising process를 위한 배경 소음이 있는 프롬프트 $z_\textrm{prompt}$가 있다고 가정한다. $$z'_\textrm{source}$$와 $$z'_\textrm{prompt}$$는 동일한 배경 소음을 가진다. 

배경 소음을 제거하기 위해 먼저 $$z'_\textrm{source}$$와 $$z'_\textrm{prompt}$$에 의한 source-aware diffusion process를 적용하고 $z_1$을 얻는다. 이 절차에서는 소스 오디오의 duration과 pitch를 활용한. 그런 다음, target-aware denoising process를 실행하여 $z_1$의 깨끗한 오디오와 깨끗한 프롬프트 $z_\textrm{prompt}$를 얻는다. 특히 이 절차에서는 소스 오디오의 음소 시퀀스, duration 시퀀스, pitch 시퀀스를 사용한다. 그 결과, NaturalSpeech 2는 배경 소음을 효과적으로 제거하는 동시에 운율 및 음색과 같은 중요한 측면을 보존할 수 있다.