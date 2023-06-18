---
title: "[논문리뷰] DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion"
last_modified_at: 2023-05-22
categories:
  - 논문리뷰
tags:
  - Diffusion
  - Fine-Tuning
  - Computer Vision
  - AI
  - Google
  - NVIDIA
excerpt: "DreamPose 논문 리뷰"
use_math: true
classes: wide
---

> arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06025)] [[Page](https://grail.cs.washington.edu/projects/dreampose/)] [[Github](https://github.com/johannakarras/DreamPose)]  
> Johanna Karras, Aleksander Holynski, Ting-Chun Wang, Ira Kemelmacher-Shlizerman  
> University of Washington, UC Berkeley, Google Research, NVIDIA  
> 12 Apr 2023  

<center><img src='{{"/assets/img/dreampose/dreampose-fig1.PNG" | relative_url}}' width="90%"></center>

## Introduction
본 논문에서는 주어진 포즈 시퀀스를 사용하여 패션 사진을 사실적인 애니메이션 동영상으로 변환하는 방법인 DreamPose를 소개한다. Dreampose는 Stable Diffusion 기반의 diffusion 동영상 합성 모델이다. 하나 이상의 인간 이미지와 포즈 시퀀스가 주어지면 DreamPose는 포즈 시퀀스에 따라 입력 대상의 고품질 동영상을 생성한다.

여러모로 어려운 작업이다. 이미지 diffusion model은 인상적인 고품질 결과를 보여주었지만 동영상 diffusion model은 아직 동일한 품질의 결과를 달성하지 못했다. 더욱이 기존의 동영상 diffusion model은 시간적 일관성, motion jitter, 사실감 부족, 대상 비디오의 모션 또는 세부적인 개체 모양을 제어할 수 없다는 문제가 있다. 이것은 부분적으로는 기존 모델이 더 세밀한 제어를 제공할 수 있는 다른 컨디셔닝 신호(예: 동작)와 달리 주로 텍스트로 컨디셔닝되기 때문이다. 대조적으로 본 논문의 이미지 및 포즈 컨디셔닝 체계는 더 큰 외관 충실도와 프레임 간 일관성을 허용한다.

Dreampose는 이미 자연 이미지의 분포를 효과적으로 모델링하는 기존의 사전 학습된 이미지 diffusion model에서 finetuning된다. 이러한 모델을 사용할 때 이미지 애니메이션 작업은 컨디셔닝 신호와 일치하는 자연 이미지의 subspace를 찾는 것으로 효과적으로 단순화될 수 있다. 이를 위해 Stable Diffusion 아키텍처의 인코더와 컨디셔닝 메커니즘을 재설계하여 이미지와 포즈 컨디셔닝을 활성화한다. 또한 하나 이상의 입력 이미지에서 UNet과 VAE를 모두 finetuning하는 2단계 finetuning 방식을 제안한다.

## Background
**Diffusion model**은 품질, 다양성, 학습 안정성 측면에서 합성 task에서 GAN을 능가하는 최신 생성 모델 클래스이다. 표준 이미지 diffusion model은 정규 분포된 랜덤 noise에서 이미지를 반복적으로 복구하는 방법을 학습한다. **Latent diffusion model** (ex. Stable Diffusion)은 오토인코더의 인코딩된 latent space에서 작동하므로 최소한의 품질을 희생하면서 계산 복잡성을 절약한다. Stable Diffusion은 VAE와 Denoising UNet의 두 가지 모델로 구성된다. 오토인코더는 프레임 $x$를 간결한 latent 표현 $z = \mathcal{E} (x)$로 추출하는 인코더 $\mathcal{E}$와 latent 표현 $x' = \mathcal{D} (z)$에서 이미지를 재구성하는 디코더 $\mathcal{D}$로 구성된다. 학습하는 동안, latent feature $z$는 랜덤 noise와 구별할 수 없는 noisy feature $$\tilde{z}_T$$를 생성하기 위해 결정론적 Gaussian process에 의해 $T$ timestep으로 diffuse된다. 원본 이미지를 복구하기 위해 각 timestep $$t \in \{1, \cdots, T\}$$에 해당하는 latent feature의 noise를 반복적으로 예측하도록 시간으로 컨디셔닝된 UNet을 학습된다. UNet $\epsilon_\theta$의 목적 함수는 다음과 같다.

$$
\begin{equation}
L_{DM} = \mathbb{E}_{z, \epsilon \in \mathcal{N}(0,1)} [\| \epsilon - \epsilon_\theta (\tilde{z}_t, t, c) \|_2^2]
\end{equation}
$$

여기서 $c$는 컨디셔닝 정보의 임베딩을 나타낸다. 마지막으로, 예측된 latent $z'$는 예측된 이미지 $x' = \mathcal{D}(z')$를 복구하도록 디코딩된다.

**Classifier-free guidance**는 암시적 classifier를 통해 예측된 noise 분포를 조건부 분포로 푸시하는 샘플링 메커니즘이다. 이는 랜덤한 확률로 실제 컨디셔닝 입력을 null 입력($\emptyset$)으로 대체하는 학습 체계인 dropout에 의해 달성된다. Inference하는 동안 조건부 예측은 스칼라 가중치 $s$를 사용하여 unconditional한 예측을 조건부로 guide하는 데 사용된다.

$$
\begin{equation}
\epsilon_\theta = \epsilon_\theta(\tilde{z}_t, t, \emptyset) + s \cdot (\epsilon_\theta (\tilde{z}_t, t, c) - \epsilon_\theta (\tilde{z}_t, t, \emptyset))
\end{equation}
$$

## Method
본 논문의 방법은 단일 이미지와 포즈 시퀀스에서 사실적인 애니메이션 동영상을 만드는 것을 목표로 한다. 이를 위해 패션 동영상 컬렉션에서 사전 학습된 Stable Diffusion 모델을 finetuning한다. 여기에는 추가 컨디셔닝 신호(이미지 및 포즈)를 받고 동영상으로 볼 수 있는 시간적으로 일관된 콘텐츠를 출력하기 위해 Stable Diffusion의 아키텍처를 조정하는 작업이 포함된다.

### 1. Overview
입력 이미지 $x_0$와 포즈 $$\{p_1, \cdots, p_N\}$$이 주어지면 동영상 $$\{x_1', \cdots, x_N'\}$$을 생성한다. 여기서 $x_i'$는 입력 포즈 $p_i$에 해당하는 $i$번째 예측 프레임이다. 본 논문의 방법은 입력 이미지와 일련의 포즈로 컨디셔닝되는 사전 학습된 latent diffusion model에 의존한다. Inference 시 표준 diffusion 샘플링 절차를 통해 각 프레임을 독립적으로 생성한다. 균일하게 분포된 Gaussian noise에서 시작하여 diffusion model은 두 컨디셔닝 신호로 반복적으로 쿼리되어 noisy latent를 그럴듯한 추정치로 점진적으로 제거한다. 마지막으로, 예측된 denoised latent $z_i'$는 예측된 동영상 프레임 $x_i' = \mathcal{D}(z_i')$를 생성하기 위해 디코딩된다.

### 2. Architecture
DreamPose 모델은 이미지 애니메이션을 위해 원래의 text-to-image Stable Diffusion 모델을 수정하고 finetuning하는 포즈 및 이미지로 컨디셔닝된 이미지 생성 모델이다. 이미지 애니메이션의 목적은 


1. 제공된 입력 이미지에 대한 충실도
2. 시각적 품질
3. 생성된 프레임 전체에서 시간적 안정성

을 포함한다. 이와 같이 DreamPose에는 전체 구조, 개인 신원(identity), 옷의 세밀한 디테일을 캡처하는 이미지 컨디셔닝 메커니즘과 대상 포즈에서 출력 이미지를 효과적으로 컨디셔닝하는 동시에 독립적으로 샘플링된 출력 프레임 간의 시간적 일관성을 가능하게 하는 방법이 필요하다. 아키텍처의 다이어그램은 아래 그림과 같다.

<center><img src='{{"/assets/img/dreampose/dreampose-fig2.PNG" | relative_url}}' width="100%"></center>

#### Split CLIP-VAE Encoder
[InstructPix2Pix](https://arxiv.org/abs/2211.09800)와 같은 많은 이전 연구에서 이미지 컨디셔닝 신호는 종종 denoising U-Net에 대한 입력 noise와 concat된다. 이것은 원하는 출력 이미지와 공간적으로 정렬된 신호를 컨디셔닝하는 데 효과적이지만, Dreampose의 경우 네트워크는 특히 입력 이미지와 공간적으로 정렬되지 않은 이미지를 생성하는 것을 목표로 한다. 따라서 저자들은 이미지 컨디셔닝을 위한 다른 접근 방식을 탐색하였다. 특히 사전 학습된 CLIP 이미지와 VAE 인코더의 인코딩된 정보를 결합하는 맞춤형 컨디셔닝 어댑터로 CLIP 텍스트 인코더를 대체하여 이미지 컨디셔닝을 구현한다.

사전 학습된 네트워크에서 finetuning할 때 중요한 목표는 입력 신호를 원래 네트워크 학습에 사용된 신호와 최대한 유사하게 만들어 학습 기울기를 가능한 한 의미 있게 만드는 것이다. 이렇게 하면 finetuning 중 네트워크 성능의 회귀 또는 noise 기울기에서 발생할 수 있는 학습된 prior 값의 손실을 방지하는 데 도움이 된다. 이러한 이유로 대부분의 diffusion 기반 finetuning 체계는 모든 원래 컨디셔닝 신호를 유지하고 새로운 컨디셔닝 신호와 상호 작용하는 네트워크 가중치를 0으로 초기화한다.

Stable Diffusion이 텍스트 프롬프트의 CLIP 임베딩으로 컨디셔닝되고 CLIP이 텍스트와 이미지를 공유 임베딩 space로 인코딩한다는 점을 감안할 때 CLIP 컨디셔닝을 컨디셔닝 이미지에서 파생된 임베딩으로 간단히 대체하는 것이 자연스러워 보일 수 있다. 이것은 이론적으로 원래 아키텍처에 매우 작은 변화를 일으키고 최소한의 finetuning으로 이미지 컨디셔닝을 허용하지만 실제로는 CLIP 이미지 임베딩만으로는 컨디셔닝 이미지에서 세밀한 디테일을 캡처하기에 충분하지 않다. 그래서 대신 Stable Diffusion의 VAE에서 인코딩된 latent 임베딩을 추가로 입력한다. 이러한 latent 임베딩을 컨디셔닝으로 추가하면 diffusion model의 출력 도메인과 일치하는 추가 이점이 있다.

아키텍처는 기본적으로 컨디셔닝 신호로 VAE latent를 지원하지 않기 때문에 CLIP과 VAE 임베딩을 결합하여 네트워크의 일반적인 cross-attention 연산에 사용되는 하나의 임베딩을 생성하는 어댑터 모듈 $\mathcal{A}$를 추가한다. 이 어댑터는 두 신호를 함께 혼합하고 denoising U-Net의 cross-attention 모듈에서 예상하는 일반적인 모양으로 출력을 변환한다. 앞서 언급했듯이 학습에서 네트워크의 충격을 완화하기 위해 처음에 VAE 임베딩에 해당하는 가중치는 0으로 설정되어 네트워크가 CLIP 임베딩으로만 학습을 시작한다. 최종 이미지 컨디셔닝 신호 $c_I$를 다음과 같이 정의한다.

$$
\begin{equation}
c_I = \mathcal{A}(c_\textrm{CLIP}, c_\textrm{VAE})
\end{equation}
$$

#### Modified UNet
이미지 컨디셔닝과 달리 포즈 컨디셔닝은 이미지와 정렬된다. 이와 같이 noisy latent $$\tilde{z}_i$$를 타겟 포즈 표현 $c_p$와 concat한다. 포즈의 noise를 설명하고 생성된 프레임의 시간적 일관성을 최대화하기 위해 $c_p$를 5개의 연속 포즈 프레임으로 구성하도록 설정한다. 

$$
\begin{equation}
c_p = \{p_{i-2}, p_{i-1}, p_i, p_{i+1}, p_{i+2}\}
\end{equation}
$$

개별 포즈로 네트워크를 학습하면 프레임 간에 jitter되는 경향이 있지만 일련의 연속 포즈로 학습하면 전반적인 움직임의 부드러움과 시간적 일관성이 증가한다. 구조적으로 0으로 초기화된 10개의 추가 입력 채널을 받아들이도록 UNet 입력 레이어를 수정하고 noisy latent에 해당하는 원래 채널은 사전 학습된 가중치에서 수정되지 않는다.

### 3. Finetuning
초기화를 위해 별도의 사전 학습된 체크포인트에서 로드되는 CLIP 이미지 인코더를 제외하고 수정되지 않은 Stable Diffusion 레이어는 사전 학습된 text-to-image Stable Diffusion 체크포인트에서 초기화된다. 이전에 언급한 바와 같이 새로운 layer는 초기에 새로운 컨디셔닝 신호가 네트워크 출력에 기여하지 않도록 초기화된다.

<center><img src='{{"/assets/img/dreampose/dreampose-fig4.PNG" | relative_url}}' width="80%"></center>
<br>
초기화 후 DreamPose는 두 단계로 finetuning된다. 첫 번째 단계는 입력 이미지 및 포즈와 일치하는 프레임을 합성하기 위해 전체 학습 데이터셋에서 UNet과 어댑터 모듈을 finetuning한다. 두 번째 단계에서는 하나 이상의 주제별 입력 이미지에서 UNet과 어댑터 모듈, VAE 디코더를 finetuning하여 base model을 개선하여 inference에 사용되는 주제별 사용자 정의 모델을 생성한다.

다른 이미지 조건부 확산 방법과 유사하게 샘플별 finetuning은 입력 이미지의 사람과 옷의 identity를 보존하고 프레임 전체에서 일관된 모양을 유지하는 데 필수적이다. 그러나 단순히 단일 프레임 및 포즈 쌍에 대한 학습은 texture-sticking과 같은 출력 동영상의 아티팩트를 빠르게 발생시킨다. 이를 방지하기 위해 random crop을 추가하는 등 각 단계에서 이미지-포즈 쌍을 늘린다.

또한 VAE 디코더를 finetuning하는 것이 합성된 출력 프레임에서 더 선명하고 사실적인 디테일을 복구하는 데 중요하다. (아래 그림 참고)

<center><img src='{{"/assets/img/dreampose/dreampose-fig5.PNG" | relative_url}}' width="80%"></center>

### 4. Pose and Image Classifier-Free Guidance
Inference 시 단일 입력 이미지와 주제별 모델을 사용하는 일련의 포즈에서 프레임별로 동영상을 생성한다. 이중 classifier-free guidance를 사용하여 inference 중에 이미지 컨디셔닝 $c_I$와 포즈 컨디셔닝 $c_p$의 강도를 조절한다. 이중 classifier-free guidance 방정식은 출력 이미지가 입력 이미지 $c_I$와 입력 포즈 $c_p$와 각각 얼마나 유사한지를 결정하는 두 개의 guidance 가중치 $s_I$와 $s_p$에 의해 제어되도록 수정된다.

$$
\begin{aligned}
\epsilon_\theta(z_t, c_I, c_p) &= \epsilon_\theta (z_t, \emptyset, \emptyset) \\
&+ s_I (\epsilon_\theta (z_t, c_I, \emptyset) - \epsilon_\theta (z_t, \emptyset, \emptyset)) \\
&+ s_p (\epsilon_\theta (z_t, c_I, c_p) - \epsilon_\theta (z_t, c_I, \emptyset))
\end{aligned}
$$

<center><img src='{{"/assets/img/dreampose/dreampose-fig7.PNG" | relative_url}}' width="70%"></center>
<br>
위 그림에서는 classifier-free guidance 가중치 $(s_I, s_p)$에 따른 효과를 보여준다. 큰 $s_I$는 입력 이미지에 대한 높은 외관 충실도를 보장하는 반면, 큰 $s_p$는 입력 포즈에 대한 정렬을 보장한다. 포즈 및 이미지 guidance를 강화하는 것 외에도 분리된 classifier-free guidance는 주제별 finetuning 후 하나의 입력 포즈에 대한 overfitting을 방지한다.

## Experiments
- 데이터셋: UBC Fashion dataset
- Implementation Details
  - 512$\times$512에서 NVIDIA A100 GPU 2개로 학습
  - 첫번째 단계: 전체 학습 데이터셋
    - 5 epoch, learning rate $5 \times 10^{-6}$
    - batch size: 16 (4 gradient accumulation step)
    - Dropout: 포즈 입력 5%, 이미지 입력 5%
  - 두번째 단계
    - 특정 샘플 프레임: 500 step, learning rate $1 \times 10^{-5}$
    - VAE 디코더: 1500 step, learning rate $5 \times 10^{-5}$
  - Inference는 PNDM sampler를 사용 (100 step)

다음은 다양한 입력 프레임과 포즈들에 대한 결과 샘플들이다.

<center><img src='{{"/assets/img/dreampose/dreampose-fig3.PNG" | relative_url}}' width="100%"></center>

### 1. Comparisons
####  Quantitative Analysis
다음은 DreamPose를 MRAA와 TPSMM과 정량적으로 비교한 결과이다.

<center><img src='{{"/assets/img/dreampose/dreampose-table1.PNG" | relative_url}}' width="50%"></center>

#### Qualitative Analysis
다음은 DreamPose를 MRAA와 TPSMM과 정성적으로 비교한 결과이다.

<center><img src='{{"/assets/img/dreampose/dreampose-fig6.PNG" | relative_url}}' width="70%"></center>
<br>
다음은 DreamPose를 PIDM과 정성적으로 비교한 결과이다.

<center><img src='{{"/assets/img/dreampose/dreampose-fig9.PNG" | relative_url}}' width="75%"></center>

### 2. Ablation Studies
저자들은 ablation study를 위해 4가지 버전의 모델을 비교하였다.

1. **CLIP**: CLIP-VAE 인코더 대신 사전 학습된 CLIP 이미지 인코더를 사용한 버전
2. **No-VAE-FT**: VAE 디코더에 대한 finetuning을 하지 않은 버전
3. **1-pose**: 5개의 연속된 포즈 대신 하나의 포즈만 noise에 concat한 버전
4. **full**: CLIP-VAE 인코더, VAE 디코더 finetuning, 5개의 연속된 포즈 입력을 모두 사용한 버전

#### Quantitative Comparison
다음은 각 버전에 대한 정량적 비교 결과이다. 

<center><img src='{{"/assets/img/dreampose/dreampose-table2.PNG" | relative_url}}' width="50%"></center>

#### Qualitative Comparison
다음은 각 버전에 대한 정성적 비교 결과이다. 

<center><img src='{{"/assets/img/dreampose/dreampose-fig8.PNG" | relative_url}}' width="100%"></center>

### 3. Multiple Input Images
Dreampose는 하나의 입력 이미지에 대하여 고품질의 결과를 생성한다. Dreampose는 피사체에 대한 임의의 입력 이미지 개수로 학습할 수 있다. 다음은 1, 3, 5, 7개의 입력 이미지로 학습한 결과이다.

<center><img src='{{"/assets/img/dreampose/dreampose-fig10.PNG" | relative_url}}' width="80%"></center>
<br>
피사체의 추가 입력 이미지는 품질과 일관성을 높인다. 

## Limitations
<center><img src='{{"/assets/img/dreampose/dreampose-fig11.PNG" | relative_url}}' width="60%"></center>
<br>
위 그림에서는 실패 사례를 보여준다. 드문 경우지만 대상 포즈가 뒤를 향할 때 팔다리가 옷 속으로 사라지고(왼쪽), hallucinate feature(중간)와 방향 불일치(오른쪽)가 관찰된다. 또한 Dreampose는 대부분의 단순한 패턴의 옷에서 사실적인 결과를 생성하지만 일부 결과는 크고 복잡한 패턴에서 약간의 깜박임 동작을 보인다. 마지막으로 다른 diffusion model과 마찬가지로 finetuning 및 inference 시간이 GAN 또는 VAE에 비해 느리다. 특정 피사체에 대한 모델 finetuning은 프레임당 18초의 렌더링 시간 외에 UNet의 경우 약 10분, VAE 디코더의 경우 약 20분이 소요된다.