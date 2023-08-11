---
title: "[논문리뷰] NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality"
last_modified_at: 2023-08-11
categories:
  - 논문리뷰
tags:
  - Transformer
  - Text-to-Speech
  - Audio and Speech Processing
  - AI
  - Microsoft
excerpt: "NaturalSpeech 논문 리뷰"
use_math: true
classes: wide
---

> arXiv 2021. [[Paper](https://arxiv.org/abs/2205.04421)] [[Page](https://speechresearch.github.io/naturalspeech/)] [[Github](https://github.com/microsoft/NeuralSpeech)]  
> Xu Tan, Jiawei Chen, Haohe Liu, Jian Cong, Chen Zhang, Yanqing Liu, Xi Wang, Yichong Leng, Yuanhao Yi, Lei He, Frank Soong, Tao Qin, Sheng Zhao, Tie-Yan Liu  
> Microsoft Research Asia & Microsoft Azure Speech  
> 9 May 2022  

## Introduction
TTS는 텍스트에서 이해하기 쉽고 자연스러운 음성을 합성하는 것을 목표로 하며 딥러닝의 발달로 인해 최근 몇 년 동안 급속한 발전을 이루었다. 신경망 기반 TTS는 CNN/RNN 기반 모델에서 Transformer 기반 모델로, 기본 생성 모델 (autoregressive)에서 보다 강력한 모델(VAE, GAN, flow, diffusion)로, 계단식 음향 모델 (acoustic model)/보코더에서 완전한 end-to-end 모델로 발전했다.

인간 수준의 품질로 TTS 시스템을 구축하는 것은 항상 음성 합성 실무자들의 꿈이었다. 현재 TTS 시스템은 높은 음성 품질을 달성하지만 사람이 녹음한 것과 비교할 때 여전히 품질 격차가 있다. 이 목표를 추구하려면 몇 가지 질문에 답해야 한다.

1. TTS에서 인간 수준의 품질을 정의하는 방법은 무엇인가?
2. TTS 시스템이 인간 수준의 품질을 달성했는지 여부를 판단하는 방법은 무엇인가? 
3. 인간 수준의 품질을 달성하기 위해 TTS 시스템을 구축하는 방법은 무엇인가?

본 논문에서는 이러한 TTS의 문제점에 대해 포괄적인 연구를 수행한다. 먼저 통계적이고 측정 가능한 방법을 기반으로 TTS의 인간 수준 품질에 대한 공식적인 정의를 제공한다. 그런 다음 가설 테스트를 통해 TTS 시스템이 인간 수준의 품질을 달성했는지 여부를 판단하기 위한 몇 가지 가이드라인을 소개한다. 이 판단 방법을 사용하여 이전의 여러 TTS 시스템이 이를 달성하지 못했다는 것을 발견했다.

본 논문에서는 녹음 품질 격차를 해소하고 인간 수준의 품질을 달성하기 위해 **NaturalSpeech**라는 파형 생성 시스템에 대한 완전한 end-to-end text-to-waveform을 추가로 개발한다. 특히, 이미지/동영상/파형 생성에서 영감을 받아 VAE를 활용하여 고차원 음성 $x$를 연속적인 프레임 레벨 표현 (posterior) $q(z \vert x)$로 압축하여 파형 $p(x \vert z)$을 재구성하는 데 사용한다. 해당 prior $p(z \vert y)$는 텍스트 시퀀스 $y$에서 얻는다. 음성의 posterior가 텍스트의 prior보다 더 복잡하다는 점을 고려하여 $p(z \vert y) \rightarrow p(x \vert z)$를 통해 텍스트를 음성으로 합성할 수 있도록 posterior와 prior를 서로 최대한 가깝게 일치시키는 여러 모듈을 설계한다.

- 음소 인코더에서 대규모 사전 학습을 활용하여 음소 시퀀스에서 더 나은 표현을 추출한다.
- Duration predictor와 업샘플링 레이어로 구성된 완전히 differentiable durator를 활용하여 duration 모델링을 개선한다.
- Prior $p(z \vert y)$를 더욱 향상시키고 posterior $q(z \vert x)$의 복잡성을 줄이기 위해 flow model을 기반으로 양방향 prior/posterior 모듈을 설계한다.
- 파형 재구성에 필요한 posterior 복잡도를 줄이기 위해 메모리 기반 VAE를 제안한다.

이전 TTS 시스템과 비교할 때 NaturalSpeech는 몇 가지 장점이 있다. 

1. 학습-inference 불일치를 줄인다. 이전의 계단식 음향 모델/보코더 파이프라인과 명시적인 duration 예측에서 mel-spectrogram과 duration은 보코더와 mel-spectrogram 디코더를 학습시키는 데 ground-truth가 사용되는 반면 예측된 값은 inference에 사용되기 때문에 학습-inference 불일치로 인해 어려움을 겪는다. NaturalSpeech의 완전한 end-to-end text-to-waveform 생성과 differentiable durator는 학습-inference 불일치를 방지할 수 있다. 
2. 일대다 매핑 문제를 완화한다. 하나의 텍스트 시퀀스는 다양한 변형 정보 (ex. pitch, duration, 속도, 일시 중지, 운율 등)가 있는 여러 음성 발화에 해당할 수 있다. Pitch/duration을 예측하기 위해 variance adaptor만 사용하는 이전 연구들은 일대다 매핑 문제를 잘 처리할 수 없다. NaturalSpeech의 메모리 기반 VAE와 양방향 prior/posterior는 posterior의 복잡성을 줄이고 prior를 향상시켜 일대다 매핑 문제를 완화할 수 있다. 
3. 표현 능력을 향상시킨다. 이전 모델은 음소 시퀀스에서 좋은 표현을 추출하고 음성의 복잡한 데이터 분포를 학습할 만큼 강력하지 않다. 본 논문의 대규모 음소 사전 학습과 flow model, VAE와 같은 강력한 생성 모델은 더 나은 텍스트 표현과 음성 데이터 분포를 학습할 수 있다.

저자들은 NaturalSpeech 시스템의 음성 품질을 측정하기 위해 널리 채택된 LJSpeech 데이터셋에 대한 실험적 평가를 수행하였다. 제안된 판단 가이드라인에 따라 NaturalSpeech는 MOS와 CMOS 측면에서 인간 녹음과 유사한 품질을 달성한다. 구체적으로, NaturalSpeech에 의해 생성된 음성은 Wilcoxon signed rank test에서 $p$-level $p \gg 0.05$로 녹음과 비교하여 -0.01 CMOS를 달성한다. 이는 NaturalSpeech가 녹음과 통계적으로 유의한 차이 없이 음성을 생성할 수 있음을 보여준다.

## Definition and Judgement of Human-Level Quality in TTS
### 1. Definition of Human-Level Quality
저자들은 통계적이고 측정 가능한 방식으로 인간 수준의 품질을 정의하였다.

**정의**: TTS 시스템에서 생성된 음성의 품질 점수와 테스트셋에 있는 해당 사람 녹음의 품질 점수 간에 통계적으로 유의미한 차이가 없으면 이 TTS 시스템은 이 테스트셋에서 사람 수준의 품질을 달성한다.

TTS 시스템이 테스트셋에서 인간 수준의 품질을 달성한다고 주장한다고 해서 TTS 시스템이 인간을 능가하거나 대체할 수 있다는 의미는 아니지만 이 TTS 시스템의 품질은 통계적으로 이 테스트셋에서 인간의 기록과 구별할 수 없다.

### 2. Judgement of Human-Level Quality
#### 판단 가이드라인
PESQ, STOI, SI-SDR과 같이 생성된 음성과 사람이 녹음한 것 사이의 품질 차이를 측정하기 위한 몇 가지 객관적인 메트릭이 있지만 TTS에서 인식 품질을 측정하기에는 신뢰할 수 없다. 따라서 음성 품질을 측정하기 위해 주관적인 평가를 사용한다. 이전 연구들에서는 일반적으로 생성된 음성과 녹음을 비교하기 위해 5점 (1 ~ 5)의 평균 의견 점수 (MOS)를 사용했다. 그러나 MOS는 음성 품질의 차이에 충분히 민감하지 않다. 왜냐하면 심사위원은 쌍을 이루는 비교 없이 두 시스템에서만 각 문장의 품질을 평가하기 때문이다. 따라서 평가 척도로 7점 (-3 ~ 3)의 비교 평균 의견 점수 (CMOS)를 선택하고 각 심사위원은 두 시스템의 샘플을 머리로 비교하여 음성 품질을 측정한다. 또한 저자들은 Wilcoxon signed rank test를 수행하여 CMOS 평가 측면에서 두 시스템이 크게 다른지 여부를 측정하였다.

따라서 인간 수준의 품질에 대한 판단 가이드라인은 다음과 같다. 

1. TTS 시스템의 각 발화와 인간 녹음은 모국어를 사용하는 20명 이상의 심사위원이 나란히 듣고 비교해야 한다. 각 시스템에서 최소 50개의 테스트 발화가 판단에 사용되어야 한다. 
3. TTS 시스템에 의해 생성된 음성은 평균 CMOS가 0에 가깝고 Wilcoxon signed rank test의 $p$-level이 $p > 0.05$를 만족하는 경우에만 사람이 녹음한 음성과 통계적으로 유의미한 차이가 없다.

#### 기존 TTS 시스템에 대한 판단
저자들은 이러한 가이드라인을 기반으로 현재 TTS 시스템이 LJSpeech 데이터셋에서 인간 수준의 품질을 달성할 수 있는지 여부를 테스트하였다. 저자들이 테스트한 시스템은 다음과 같다. 

1. FastSpeech 2 + HiFiGAN
2. Glow-TTS + HiFiGAN
3. GradTTS + HiFiGAN
4. VITS

저자들은 이러한 모든 시스템의 결과를 자체적으로 재생산하여 원본 논문의 품질과 일치하거나 심지어 능가할 수 있다 (HiFiGAN 보코더는 더 나은 합성 품질을 위해 예측된 mel-spectrogram에서 fine-tuning됨). MOS와 CMOS 평가를 위해 각각 20명의 심사위원이 있는 50개의 테스트 발화를 사용한다. 

<center><img src='{{"/assets/img/naturalspeech/naturalspeech-table1.PNG" | relative_url}}' width="80%"></center>
<br>
위 표에서 볼 수 있듯이 현재의 TTS 시스템은 녹음으로 가까운 MOS를 달성할 수 있지만 Wilcoxon signed rank test가 $p$-level $p \ll 0.05$에서 기록과 큰 CMOS 간격을 가지며 이는 사람의 녹음과 통계적으로 유의미한 차이를 보여준다. 

## Description of NaturalSpeech System
### 1. Design Principle
VQ-VAE를 사용하여 고차원 이미지를 저차원 표현으로 압축하여 생성을 용이하게 하는 이미지/동영상 생성에서 영감을 받아 VAE를 활용하여 고차원 음성 $x$를 프레임 레벨 표현 $z$로 압축한다. 즉, $z$는 posterior 분포 $q(z \vert x)$에서 샘플링된다. 이는 파형 $p(x \vert z)$을 재구성하는 데 사용된다. VAE의 일반 공식에서 prior $p(z)$는 표준 등방성 다변량 가우시안으로 선택된다. TTS의 입력 텍스트에서 조건부 파형 생성을 활성화하기 위해 음소 시퀀스 $y$에서 $z$를 예측한다. 즉, $z$는 예측된 prior 분포 $p(z \vert y)$에서 샘플링된다. $q(z \vert x)$와 $p(z \vert y)$ 모두에 전파되는 기울기를 사용하여 VAE와 prior 예측을 공동으로 최적화한다. Evidence lower bound (ELBO)에서 파생된 loss function은 파형 재구성 loss

$$
\begin{equation}
- \log p(x \vert z)
\end{equation}
$$

와 posterior $q(z \vert x)$와 prior $p(z \vert y)$ 사이의 KL divergence loss로

$$
\begin{equation}
\textrm{KL} [q(z \vert x) \| p (z \vert y)]
\end{equation}
$$

구성된다.

음성의 posterior가 텍스트의 prior보다 더 복잡하다는 점을 고려하여 텍스트를 파형 생성에 최대한 가깝게 일치시키기 위해 posterior를 단순화하고 prior를 향상시키는 여러 모듈을 설계한다. 

1. 더 나은 prior 예측을 위해 음소 시퀀스의 좋은 표현을 학습하기 위해 음소 시퀀스에서 masked language modeling을 사용하여 대규모 텍스트 코퍼스에서 음소 인코더를 사전 학습한다. 
2. Posterior는 프레임 레벨에 있고 음소 prior는 음소 레벨이기 때문에 duration 차이를 메우기 위해 음소 prior를 duration에 따라 확장해야 한다. Duration 모델링을 개선하기 위해 differentiable durator를 활용한다.
3. Prior를 강화하거나 posterior를 단순화하기 위해 양방향 prior/posterior 모듈을 설계한다
4. 파형을 재구성하는 데 필요한 posterior 복잡도를 줄이기 위해 Q-K-V attention을 통해 메모리 뱅크를 활용하는 메모리 기반 VAE를 제안한다

### 2. Phoneme Encoder
음소 인코더 $\theta_\textrm{pho}$는 음소 시퀀스 $y$를 입력으로 받아 음소 hidden 시퀀스를 출력한다. 음소 인코더의 표현 능력을 향상시키기 위해 대규모 음소 사전 학습을 수행한다. 이전 연구들은 문자/단어 레벨에서 사전 학습을 수행하고 사전 학습된 모델을 음소 인코더에 적용하면 불일치가 발생하고, 음소 사전 학습을 직접 사용하는 작업은 음소 vocabulary의 크기가 너무 작아 용량 제한이 있다. 

<center><img src='{{"/assets/img/naturalspeech/naturalspeech-fig2c.PNG" | relative_url}}' width="47%"></center>
<br>
이러한 문제를 방지하기 위해 위 그림과 같이 음소와 상위 음소 (인접한 음소가 함께 병합됨)를 모델의 입력으로 사용하는 혼합 음소 사전 학습을 활용한다. Masked language modeling을 사용할 때 일부 상위 음소 토큰과 해당 음소 토큰을 무작위로 마스킹하고 마스킹된 음소와 상위 음소를 동시에 예측한다. 혼합 음소 사전 학습 후 사전 학습된 모델을 사용하여 TTS 시스템의 음소 인코더를 초기화한다.

### 3. Differentiable Durator
<center><img src='{{"/assets/img/naturalspeech/naturalspeech-fig2a.PNG" | relative_url}}' width="22%"></center>
<br>
differentiable duratior $\theta_\textrm{dur}$는 음소 hidden 시퀀스를 입력으로 취하고 위 그림과 같이 프레임 레벨에서 prior 분포 시퀀스를 출력한다. Prior 분포를 

$$
\begin{equation}
p(z' \vert y; \theta_\textrm{pho}, \theta_\textrm{dur}) = p(z' \vert y; \theta_\textrm{pri}) \\
\textrm{where} \quad \theta_\textrm{pri} = [\theta_\textrm{pho}, \theta_\textrm{dur}]
\end{equation}
$$

로 표시한다. Differentiable durator $\theta_\textrm{dur}$는 여러 모듈로 구성된다.

1. 각 음소의 duration을 예측하기 위해 음소 인코더를 기반으로 하는 duration predictor
2. 음소 hidden 시퀀스를 미분 가능한 방식으로 음소 레벨에서 프레임 레벨로 확장하기 위해 projection 행렬을 학습하기 위해 예측된 duration을 활용하는 학습 가능한 업샘플링 레이어
3. Prior 분포 $p(z' \vert y; \theta_\textrm{pri})$의 평균과 분산을 계산하기 위해 확장된 hidden 시퀀스의 두 개의 추가 linear layer. 

TTS 모델과 함께 duration 예측, 학습 가능한 업샘플링 레이어, 평균/분산 linear layer를 완전히 미분 가능한 방식으로 최적화하여 이전 duration 예측에서 학습-inference 불일치를 줄일 수 있다. Ground-truth duration은 학습에 사용되고 예측 duration은 inference에 사용된다. 하드한 확장 대신 소프트하고 유연한 방식으로 duration을 더 잘 사용하므로 부정확한 duration 예측의 부작용이 완화된다.

### 4. Bidirectional Prior/Posterior
<center><img src='{{"/assets/img/naturalspeech/naturalspeech-fig2b.PNG" | relative_url}}' width="55%"></center>
<br>
위 그림과 같이 음성 시퀀스에서 얻은 posterior와 음소 시퀀스에서 얻은 prior 사이에 정보 격차가 있기 때문에 prior $p(z' \vert y; \theta_\textrm{pri})$의 용량을 향상시키거나 posterior encoder $\phi$에 대해 posterior $q(z \vert x; \phi)$의 복잡성을 줄이기 위해 양방향 prior/posterior 모듈을 설계한다. 양방향 prior/posterior 모듈 $\theta_\textrm{bpp}$로 flow model을 선택한다. 이는 최적화하기 쉽고 가역성이 좋기 때문이다.

#### Reduce Posterior $q(z \vert x; \phi)$ with Backward Mapping $f^{−1}$
양방향 prior/posterior 모듈은 backward 매핑 $f^{−1} (z; \theta_\textrm{bpp})$을 통해 $q(z \vert x; \phi)$에서 $q(z' \vert x; \phi, \theta_\textrm{bpp})$로 posterior 복잡도를 줄일 수 있다. 목적 함수는 다음과 같이 KL divergence loss를 사용하여 단순화된 posterior $q(z' \vert x; \phi, \theta_\textrm{bpp})$를 prior $p(z' \vert y; \theta_\textrm{pri})$와 일치시킨다.

$$
\begin{aligned}
&\mathcal{L}_\textrm{bwd} (\phi, \theta_\textrm{bpp}, \theta_\textrm{pri}) \\
&\;= \textrm{KL} [q (z' \vert x; \phi, \theta_\textrm{bpp}) \;\|\; p (z' \vert y; \theta_\textrm{pri})] \\
&\;= \int q (z' \vert x; \phi, \theta_\textrm{bpp}) \cdot \log \frac{q (z' \vert x; \phi, \theta_\textrm{bpp})}{p (z' \vert y; \theta_\textrm{pri})} dz' \\
&\;= \int q (z \vert x; \phi) \vert \textrm{det} \frac{\partial f^{-1} (z; \theta_\textrm{bpp})}{\partial z} \vert^{-1} \cdot \log \frac{q (z \vert x; \phi) \vert \textrm{det} \frac{\partial f^{-1} (z; \theta_\textrm{bpp})}{\partial z} \vert^{-1}}{p (f^{-1} (z; \theta_\textrm{bpp}) \vert y; \theta_\textrm{pri})} \cdot \vert \textrm{det} \frac{\partial f^{-1} (z; \theta_\textrm{bpp})}{\partial z} \vert dz \\
&\;= \int q (z \vert x; \phi) \cdot \log \frac{q (z \vert x; \phi)}{p (f^{-1} (z; \theta_\textrm{bpp}) \vert y; \theta_\textrm{pri}) \vert \textrm{det} \frac{\partial f^{-1} (z; \theta_\textrm{bpp})}{\partial z} \vert} dz \\
&\;= \mathbb{E}_{z \sim q(z \vert x; \phi)} [ \log q (z \vert x; \phi) - \log (p (f^{-1} (z; \theta_\textrm{bpp}) \vert y; \theta_\textrm{pri}) \vert \textrm{det} \frac{\partial f^{-1} (z; \theta_\textrm{bpp})}{\partial z} \vert ) ]
\end{aligned}
$$

여기서 위 식의 세 번째 등식은 역함수 정리에 따른 변수 변경을 통해 얻는다. 

$$
\begin{equation}
dz' = \vert \textrm{det} \frac{\partial f^{-1} (z; \theta_\textrm{bpp})}{\partial z} \vert dz \\
q(z' \vert x; \phi, \theta_\textrm{bpp}) = q (z \vert x; \phi) \vert \textrm{det} \frac{\partial f (z'; \theta_\textrm{bpp})}{\partial z'} \vert = q (z \vert x; \phi) \vert \textrm{det} \frac{\partial f^{-1} (z; \theta_\textrm{bpp})}{\partial z} \vert^{-1}
\end{equation}
$$

#### Enhance Prior $p(z' \vert y; \theta_\textrm{pri})$ with Forward Mapping $f$
양방향 prior/posterior 모듈은 forward 매핑 $f(z'; \theta_\textrm{bpp})$를 통해 $p(z' \vert y; \theta_\textrm{pri})$에서 $p(z \vert y; \theta_\textrm{pri}, \textrm{bpp})$로 prior의 용량을 향상시킬 수 있다. 목적 함수는 다음과 같이 KL divergence loss를 사용하여 향상된 prior $p(z \vert y; \theta_\textrm{pri}, \theta_\textrm{bpp})$를 posterior $q(z \vert x; \phi)$와 일치시킨다.

$$
\begin{aligned}
&\mathcal{L}_\textrm{fwd} (\phi, \theta_\textrm{bpp}, \theta_\textrm{pri}) \\
&\;= \textrm{KL} [p (z \vert y; \theta_\textrm{pri}, \theta_\textrm{bpp}) \;\|\; q (z \vert x; \phi)] \\
&\;= \int p (z \vert y; \theta_\textrm{pri}, \theta_\textrm{bpp}) \cdot \log \frac{p (z \vert y; \theta_\textrm{pri}, \theta_\textrm{bpp})}{q (z \vert x; \phi)} dz \\
&\;= \int p (z' \vert y; \theta_\textrm{pri}) \vert \textrm{det} \frac{\partial f (z'; \theta_\textrm{bpp})}{\partial z'} \vert^{-1} \cdot \log \frac{p (z' \vert y; \theta_\textrm{pri}) \vert \textrm{det} \frac{\partial f (z'; \theta_\textrm{bpp})}{\partial z'} \vert^{-1}}{q (f (z'; \theta_\textrm{bpp}) \vert x; \phi)} \cdot \vert \textrm{det} \frac{\partial f (z'; \theta_\textrm{bpp})}{\partial z'} \vert dz' \\
&\;= \int p (z' \vert y; \theta_\textrm{pri}) \cdot \log \frac{p (z' \vert y; \theta_\textrm{pri})}{q (f (z'; \theta_\textrm{bpp}) \vert x; \phi) \vert \textrm{det} \frac{\partial f (z'; \theta_\textrm{bpp})}{\partial z'} \vert} dz' \\
&\;= \mathbb{E}_{z' \sim p(z' \vert y; \theta_\textrm{pri})} [ \log p (z' \vert y; \theta_\textrm{pri}) - \log (q (f (z'; \theta_\textrm{bpp}) \vert x; \phi) \vert \textrm{det} \frac{\partial f (z'; \theta_\textrm{bpp})}{\partial z'} \vert ) ]
\end{aligned}
$$

여기서 위 식의 세 번째 등식은 역함수 정리에 따른 변수 변경을 통해 얻는다. 

$$
\begin{equation}
dz = \vert \textrm{det} \frac{\partial f (z'; \theta_\textrm{bpp})}{\partial z'} \vert dz' \\
p(z \vert y; \theta_\textrm{pri}, \theta_\textrm{bpp}) = p (z' \vert y; \theta_\textrm{pri}) \vert \textrm{det} \frac{\partial f^{-1} (z; \theta_\textrm{bpp})}{\partial z} \vert = p (z' \vert y; \theta_\textrm{pri}) \vert \textrm{det} \frac{\partial f (z'; \theta_\textrm{bpp})}{\partial z'} \vert^{-1}
\end{equation}
$$

Backward loss function과 forward loss function을 사용하여 학습 시 flow model의 양방향을 모두 고려하여 backward로 학습하지만 forward로 inference하는 이전 flow model에서 학습-inference 불일치를 줄일 수 있다. 

### 5. VAE with Memory
<center><img src='{{"/assets/img/naturalspeech/naturalspeech-fig2d.PNG" | relative_url}}' width="25%"></center>
<br>
원래 VAE 모델의 posterior $q(z \vert x; \phi)$는 음성 파형을 재구성하는 데 사용되므로 음소 시퀀스에서 prior보다 복잡하다. Prior 예측의 부담을 더 줄이기 위해 메모리 기반 VAE 모델을 설계하여 posterior를 단순화한다. 이 디자인의 아이디어는 파형 재구성을 위해 $z \sim q(z \vert x; \phi)$를 직접 사용하는 대신 메모리 뱅크에 attend하기 위한 query로 $z$를 사용하고 파형 재구성을 위해 attention 결과를 사용한다는 것이다 (위 그림 참조). 이런 식으로 posterior $z$는 메모리 뱅크의 attention 가중치를 결정하는 데만 사용되므로 크게 단순화된다. 메모리 VAE를 기반으로 한 파형 재구성 loss는 다음과 같다.

$$
\begin{equation}
\mathcal{L}_\textrm{rec} (\phi, \theta_\textrm{dec}) = - \mathbb{E}_{z \sim q (z \vert x; \phi)} [\log p(x \vert \textrm{Attention} (z, M, M); \theta_\textrm{dec})] \\
\textrm{Attention} (Q, K, V) = [\textrm{softmax} (\frac{QW_Q (KW_K)^\top}{\sqrt{h}}) VW_V] W_O
\end{equation}
$$

여기서 $\theta_\textrm{dec}$는 원래 파형 디코더뿐만 아니라 메모리 뱅크 $M$과 attention 파라미터 $W_Q$, $W_K$, $W_V$, $W_O$를 포함하여 메모리 메커니즘과 관련된 모델 파라미터를 포함하는 파형 디코더이다. 여기서 $M \in \mathbb{R}^{L \times h}$이고 $W_\ast \in \mathbb{R}^{h \times h}$이며, $L$은 메모리 뱅크의 크기이고 $h$는 hidden 차원이다.

### 6. Training and Inference Pipeline
파형 재구성 loss와 양방향 prior/posterior loss 외에도 더 나은 음성 품질을 위한 학습에서 전체 inference 절차를 수행하기 위해 완전히 end-to-end 최적화를 추가로 수행한다. Loss function은 다음과 같다.

$$
\begin{equation}
\mathcal{L}_\textrm{e2e} (\theta_\textrm{pri}, \theta_\textrm{bpp}, \theta_\textrm{dec}) = -\mathbb{E}_{z' \sim (z' \vert y; \theta_\textrm{pri})} [\log p(x \vert \textrm{Attention} (f(z'; \theta_\textrm{bpp}), M, M); \theta_\textrm{dec})]
\end{equation}
$$

전체 loss function은 다음과 같다.

$$
\begin{equation}
\mathcal{L} = \mathcal{L}_\textrm{bwd} (\phi, \theta_\textrm{pri}, \theta_\textrm{bpp}) + \mathcal{L}_\textrm{fwd} (\phi, \theta_\textrm{pri}, \theta_\textrm{bpp}) + \mathcal{L}_\textrm{rec} (\phi, \theta_\textrm{dec}) + \mathcal{L}_\textrm{e2e} (\theta_\textrm{pri}, \theta_\textrm{bpp}, \theta_\textrm{dec})
\end{equation}
$$

위의 loss function에 대한 몇 가지 특별한 설명이 있다. 

1. 프레임 레벨 prior 분포 $p(z' \vert y; \theta_\textrm{pri})$는 durator의 본질적으로 부정확한 duration 예측으로 인해 ground-truth 음성 프레임과 잘 정렬될 수 없기 때문에 $$\mathcal{L}_\textrm{bwd}$$와 $$\mathcal{L}_\textrm{fwd}$$에 대한 KL loss의 soft dynamic time warping (soft-DTW) 버전을 활용한다. 
2. 단순화를 위해 $$\mathcal{L}_\textrm{rec}$$와 $$\mathcal{L}_\textrm{e2e}$$의 파형 loss를 negative log-likelihood loss로 쓴다. 실제로 $$\mathcal{L}_\textrm{rec}$$는 GAN loss, feature mapping loss, mel-spectrogram loss로 구성되고, $$\mathcal{L}_\textrm{e2e}$$는 GAN loss로만 구성된다. 일치하지 않는 길이에서도 GAN loss가 여전히 잘 수행될 수 있으므로 $$\mathcal{L}_\textrm{e2e}$$에서는 soft-DTW를 사용하지 않는다. 

<center><img src='{{"/assets/img/naturalspeech/naturalspeech-fig3.PNG" | relative_url}}' width="30%"></center>
<br>
모델 학습에는 위 그림과 같이 여러 가지 기울기 흐름이 있다.

1. $$\mathcal{L}_\textrm{rec}$$ $\rightarrow$ $\theta_\textrm{dec}$ $\rightarrow$ $\phi$
2. $$\mathcal{L}_\textrm{bwd}$$ $\rightarrow$ $\theta_\textrm{dur}$ $\rightarrow$ $\theta_\textrm{pho}$
3. $$\mathcal{L}_\textrm{bwd}$$ $\rightarrow$ $\theta_\textrm{bpp}$ $\rightarrow$ $\phi$
4. $$\mathcal{L}_\textrm{fwd}$$ $\rightarrow$ $\theta_\textrm{bpp}$ $\rightarrow$ $\theta_\textrm{dur}$ $\rightarrow$ $\theta_\textrm{pho}$
5. $$\mathcal{L}_\textrm{fwd}$$ $\rightarrow$ $\phi$
6. $$\mathcal{L}_\textrm{e2e}$$ $\rightarrow$ $\theta_\textrm{dec}$ $\rightarrow$ $\theta_\textrm{bpp}$ $\rightarrow$ $\theta_\textrm{dur}$ $\rightarrow$ $\theta_\textrm{pho}$

학습 후 posterior encoder $\phi$를 버리고 $\theta_\textrm{pho}$, $\theta_\textrm{bpp}$, $\theta_\textrm{dur}$, $\theta_\textrm{dec}$만 inference에 사용한다. 학습과 inference 파이프라인은 Algorithm 1에 요약되어 있다.

<center><img src='{{"/assets/img/naturalspeech/naturalspeech-algo1.PNG" | relative_url}}' width="60%"></center>

### 7. Advantages of NaturalSpeech
NaturalSpeech 시스템의 설계가 녹음 품질 격차를 좁힐 수 있는 이유는 다음과 같다. 

1. **학습-inference 불일치 감소**: 텍스트에서 파형을 직접 생성하고 differentiable durator를 활용하여 완전히 end-to-end 최적화를 보장하여 계단식 음향 모델/보코더와 명시적 duration 예측에서 발생하는 학습-inference 불일치를 줄일 수 있다. VAE와 flow model은 본질적으로 학습-inference 불일치를 가질 수 있지만 이 문제를 완화하기 위해 backward/forward loss와 end-to-end loss를 설계하였다. 
2. **일대다 매칭 문제 완화**: 변동 정보 모델링을 위해 레퍼런스 인코더 또는 피치/에너지 추출을 사용하는 이전 방법과 비교하여 VAE의 posterior encoder $\phi$는 posterior 분포 $q(z \vert x; \phi)$에서 필요한 모든 분산 정보를 추출할 수 있는 레퍼런스 인코더처럼 작동한다. Posterior encoder와 VAE의 메모리 뱅크에서 암시적으로 학습될 수 있으므로 피치를 명시적으로 예측하지 않는다. Prior와 posterior가 서로 일치할 수 있도록 하기 위해 한편으로는 메모리 VAE와 backward 매핑으로 posterior를 단순화하고 다른 한편으로는 음소 사전 학습, differentiable durator, forward 매핑으로 prior를 향상시킨다. 따라서 일대다 매핑 문제를 크게 완화할 수 있다.
3. **표현 용량 증가**: 대규모 음소 사전 학습을 활용하여 음소 시퀀스에서 더 나은 표현을 추출하고 고급 생성 모델 (flow, VAE, GAN)을 활용하여 음성 데이터 분포를 더 잘 캡처하여 더 나은 음성 품질을 위해 TTS 모델의 표현 용량을 향상할 수 있다.

## Experiments
- 데이터셋
  - LJSpeech: 총 24시간 분량의 13,100개의 오디오 (22.05kHz)와 텍스트 전사
  - [news-crawl](https://data.statmt.org/news-crawl/en/): 2억 개의 문장
  - 텍스트/문자 시퀀스는 Phonemizer를 사용하여 음소 시퀀스로 변환
  - posterior encoder의 입력으로 선형 spectrogram을 사용
  - 선형 spectrogram은 STFT로 얻음 (FFT size = 1024, window size = 1024, hop size = 256)
  - 파형 디코더를 위한 mel-spectrogram은 선형 spectrogram에 80차원 mel-filterbanks을 적용하여 얻음
- 모델 구성
  - 음소 인코더
    - Feed-Forward Transformer (FFT) 블록 6개의 스택
    - multi-attention layer 1개, 1D convolution feedforward layer 1개, hidden size = 192
  - Differentiable durator
    - duration predictor는 3-layer convolution으로 구성
    - 양방향 prior/posterior 모듈은 4개의 연속적인 affine coupling layer를 사용
    - 양방향 학습을 안정화하기 위해 affine transform의 scaling 연산은 제거
    - affine transform의 shifting은 4-layer WaveNet (dilation rate = 1)로 예측
  - Posterior encoder
    - 16-layer WaveNet (kernel size = 5, dilation rate = 1)를 기반으로 함
  - 파형 디코더
    - 4개의 residual convolution block으로 구성
    - 각 블록은 3-layer 1D convolution
- 학습 디테일
  - 8개의 NVIDIA V100 GPU 32GB 사용
  - 동적 batch size: GPU당 8,000 음성 프레임 (hop size = 256)
  - 학습 epochs: 총 15,000
  - learning rate: 초기에 $2 \times 10^{-4}$, decay factor $\gamma$ = 0.999875
  - 학습 시작 후 1,000 epochs는 warmup stage이고 마지막 2,000 epochs는 tuning stage

### 1. Comparison with Human Recordings
다음은 NaturalSpeech와 인간의 녹음의 MOS를 비교한 표이다.

<center><img src='{{"/assets/img/naturalspeech/naturalspeech-table2.PNG" | relative_url}}' width="50%"></center>
<br>
다음은 NaturalSpeech와 인간의 녹음의 CMOS를 비교한 표이다.

<center><img src='{{"/assets/img/naturalspeech/naturalspeech-table3.PNG" | relative_url}}' width="50%"></center>

### 2. Comparison with Previous TTS Systems
다음은 NaturalSpeech와 이전 TTS 시스템들의 MOS와 CMOS를 비교한 표이다.

<center><img src='{{"/assets/img/naturalspeech/naturalspeech-table4.PNG" | relative_url}}' width="52%"></center>

### 3. Ablation Studies and Method Analyses
다음은 NaturalSpeech의 각 디자인에 대한 ablation study 결과이다.

<center><img src='{{"/assets/img/naturalspeech/naturalspeech-table5.PNG" | relative_url}}' width="37%"></center>
<br>
다음은 inference 속도를 비교한 표이다. RTF (real-time factor)는 1초의 파형을 합성하는데 걸리는 시간 (초)이다. Grad-TTS (1000)과 Grad-TTS (10)는 각각 inference애 1000 step과 10 step을 사용하였음을 나타낸다. 

<center><img src='{{"/assets/img/naturalspeech/naturalspeech-table6.PNG" | relative_url}}' width="42%"></center>