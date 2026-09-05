---
title: "[논문리뷰] One-step Latent-free Image Generation with Pixel Mean Flows"
last_modified_at: 2026-09-05
categories:
  - 논문리뷰
tags:
  - Diffusion
  - Computer Vision
  - ICML
excerpt: "pixel MeanFlow (pMF) 논문 리뷰 (ICML 2026)"
use_math: true
classes: wide
---

> ICML 2026. [[Paper](https://arxiv.org/abs/2601.22158)] [[Github](https://github.com/Lyy-iiis/pMF)]  
> Yiyang Lu, Susie Lu, Qiao Sun, Hanhong Zhao, Zhicheng Jiang, Xianbang Wang, Tianhong Li, Zhengyang Geng, Kaiming He  
> MIT | CMU  
> 29 Jan 2026  

## Introduction
본 논문에서는 1-step으로 latent 없이 이미지를 생성하기 위한 **pixel MeanFlow (pMF)**을 제안하였다. pMF는 순간 속도 $\textbf{v}$의 space에서 정의된 loss를 사용하여 average velocity field $\textbf{u}$를 학습하는 [improved MeanFlow (iMF)](https://arxiv.org/abs/2512.02012)을 따른다. 한편, [JiT](https://kimjy99.github.io/논문리뷰/jit)를 따라 pMF는 이미지 공간에서 예측을 수행한다 ($\textbf{x}$-prediction). 두 가지 방식을 모두 수용하기 위해, $\textbf{v}$, $\textbf{u}$, $\textbf{x}$를 연결하는 변환식을 도입하였다.

일반적으로 pMF는 noise가 포함된 입력을 이미지 픽셀에 직접 매핑하는 네트워크를 학습시킨다. 이는 다단계 또는 latent 기반 방식에서는 불가능한 "보는 그대로의 결과"라는 특성을 가능하게 한다. 이러한 특성 덕분에 perceptual loss를 pMF에 자연스럽게 적용할 수 있으며, 생성 품질을 더욱 향상시킬 수 있다.

pMF는 ImageNet 데이터셋에서 256$\times$256 해상도에서 2.22 FID, 512$\times$512 해상도에서 2.48 FID를 달성하였다. 또한, 픽셀 공간에서 속도를 직접 예측하는 것은 성능 저하를 초래함을 보임으로써 적절한 예측 타겟의 중요성을 입증했다. 

## Method
### 1. The Denoised Image Field
[improved MeanFlow (iMF)](https://arxiv.org/abs/2512.02012)와 [JiT](https://kimjy99.github.io/논문리뷰/jit)는 모두 $\textbf{v}$-loss를 최소화하는 것으로 볼 수 있으며, iMF는 $\textbf{u}$-prediction을, JiT는 $\textbf{x}$-prediction을 수행한다. 따라서, 본 논문에서는 $\textbf{u}$와 일반화된 형태의 $\textbf{x}$ 사이의 관계를 제시하였다.

<center><img src='{{"/assets/img/pixel-mean-flow/pixel-mean-flow-table1.webp" | relative_url}}' width="40%"></center>
<br>
[MeanFlow (MF)](https://kimjy99.github.io/논문리뷰/mean-flow)에서 정의된 average velocity field $\textbf{u}$는 다음과 같다.

$$
\begin{equation}
\textbf{u} (\textbf{z}_t, r, t) = \frac{1}{t - r} \int_r^t \textbf{v} (\textbf{z}_\tau, \tau) d \tau
\end{equation}
$$

이 velocity field는 $$p_\textrm{data}$$, $$p_\textrm{prior}$$, 시간 schedule에 의존하지만 네트워크에는 의존하지 않는 GT 값을 나타낸다. 저자들은 다음과 같이 정의된 새로운 field $$\textbf{x}(\textbf{z}_t, r, t)$$를 도입하였다.

$$
\begin{equation}
\textbf{x} (\textbf{z}_t, r, t) = \textbf{z}_t - t \cdot \textbf{u}(\textbf{z}_t, r, t)
\end{equation}
$$

이 field $\textbf{x}$는 denoising된 이미지와 유사한 역할을 한다. 이전 논문들에서 $\textbf{x}$로 언급되는 다른 값들과는 달리, 이 field $$\textbf{x}(\textbf{z}_t, r, t)$$는 두 개의 timestep $r$과 $t$로 인덱싱된다. 즉, 주어진 $$\textbf{z}_t$$에 대해 $\textbf{x}$는 $t$로만 인덱싱되는 1차원 궤적이 아니라 $(r, t)$로 인덱싱되는 2차원 field이다.

### 2. The Generalized Manifold Hypothesis
<center><img src='{{"/assets/img/pixel-mean-flow/pixel-mean-flow-fig1.webp" | relative_url}}' width="100%"></center>
<br>
위 그림은 사전 학습된 Flow Matching (FM) 모델에서 얻은 하나의 ODE 궤적을 시뮬레이션하여 $\textbf{u}$ field와 $\textbf{x}$ field를 시각화한 것이다. 그림에서 볼 수 있듯이, $\textbf{u}$는 noise와 데이터 성분을 모두 포함하고 있기 때문에 noise가 섞인 이미지로 나타난다. 반면, $\textbf{x}$는 denoising된 이미지처럼 보인다. 즉, 거의 깨끗한 이미지이거나, noise가 과도하게 제거되어 흐릿하게 보이는 이미지이다.

JiT에서는 저차원 매니폴드 상에 존재한다고 가정되는 denoising된 이미지를 직접 예측한다. 저자들이 도입한 $\textbf{x}$에 매니폴드 가설을 어떻게 일반화할 수 있는지 살펴보자. MeanFlow에서 timestep $r$은 $0 \le r \le t$를 만족하므로, 먼저 $r = t$과 $r = 0$에서 매니폴드 가설을 근사적으로 만족할 수 있음을 보이고, 그 다음 $0 < r < t$인 경우를 살펴보자.

##### Boundary case I: $r = t$
$r = t$일 때, $\textbf{u}$는 $\textbf{v}$가 된다. 즉, $$\textbf{u}(\textbf{z}_t, t, t) = \textbf{v}(\textbf{z}_t, t)$$이다. 이 경우, $\textbf{x}$는 다음과 같다.

$$
\begin{equation}
\textbf{x} (\textbf{z}_t, t, t) = \textbf{z}_t − t \cdot \textbf{v} (\textbf{z}_t, t)
\end{equation}
$$

이것은 본질적으로 JiT에서 사용되는 $\textbf{x}$-prediction 타겟이다. 직관적으로, 이 $\textbf{x}$는 JiT가 예측할 denoising된 이미지이다. Noise level이 높으면 이 denoising된 이미지는 흐릿할 수 있다. 이러한 denoising된 이미지는 저차원 매니폴드 상에 있다고 가정할 수 있다.

##### Boundary case II: $r = 0$
$r = 0$에서 $\textbf{u}$의 정의는 다음과 같다.

$$
\begin{equation}
\textbf{u}(\textbf{z}_t, 0, t) = \frac{1}{t} \int_0^t \textbf{v} (\textbf{z}_\tau, \tau) d \tau = \frac{1}{t} (\textbf{z}_t - \textbf{z}_0)
\end{equation}
$$

이를 $\textbf{x}$ 식에 대입하면 다음과 같다.

$$
\begin{equation}
\textbf{x} (\textbf{z}_t, 0, t) = \textbf{z}_0
\end{equation}
$$

즉, 이는 ODE 궤적의 종점이다. 실제 ODE 궤적의 경우 $$\textbf{z}_0 \sim p_\textrm{data}$$가 성립하는데, 이는 이미지 분포를 따라야 함을 의미한다. 따라서 $$\textbf{x}(\textbf{z}_t, 0, t)$$는 이미지 매니폴드 상에 대략적으로 위치한다고 가정할 수 있다.

##### General case: $r \in (0, t)$
$r \in (0, t)$일 때, $$\textbf{x}(\textbf{z}_t, r, t)$$는 데이터 매니폴드에서 추출한 이미지 샘플에 대응한다는 보장이 없다. 그럼에도 불구하고, 시뮬레이션 결과는 $\textbf{x}$가 denoising된 이미지처럼 보인다는 것을 보여준다. 이는 훨씬 더 noise가 많은 $\textbf{u}$와는 극명한 대조를 이룬다. 이러한 비교는 $\textbf{x}$가 $\textbf{u}$보다 신경망으로 모델링하기 더 쉬울 수 있음을 시사한다.

### 3. Algorithm
$$\textbf{x}(\textbf{z}_t, r, t)$$는 MeanFlow 네트워크의 re-parameterization을 제공한다. 구체적으로, 네트워크가 $\textbf{x}$를 직접 출력하도록 하고, 해당 velocity field $\textbf{u}$를 계산한다.

$$
\begin{equation}
\textbf{u}_\theta (\textbf{z}_t, r, t) = \frac{1}{t} (\textbf{z}_t - \textbf{x}_\theta (\textbf{z}_t, r, t))
\end{equation}
$$

위 식의 $$\textbf{u}_\theta$$를 iMF의 $\textbf{v}$-loss 식에 통합하면 loss는 다음과 같다.

$$
\begin{equation}
\mathcal{L}_\textrm{pMF} = \mathbb{E}_{t, r, \textbf{x}, \boldsymbol{\epsilon}} \| \textbf{V}_\theta - \textbf{v} \|^2 \\
\textbf{V}_\theta = \textbf{u}_\theta + (t - r) \cdot \textrm{JVP}_\textrm{sg}
\end{equation}
$$

($$\textrm{JVP}_\textrm{sg}$$는 $$\frac{d}{dt} \textbf{u}_\theta$$를 계산하는 Jacobian-vector product에 stop-gradient)

개념적으로 이는 $\textbf{x}$-prediction을 사용한 $\textbf{v}$-loss이며, $\textbf{x}$는 $\textbf{x} \rightarrow \textbf{u} \rightarrow \textbf{V}$ 관계에 의해 $\textbf{v}$-space로 변환된다.

<center><img src='{{"/assets/img/pixel-mean-flow/pixel-mean-flow-algo1.webp" | relative_url}}' width="45%"></center>

### 4. Pixel MeanFlow with Perceptual Loss
네트워크 $$\textbf{x}_\theta (\textbf{z}_t, r, t)$$는 noisy한 입력 $$\textbf{z}_t$$를 denoising된 이미지로 직접 매핑한다. 이를 통해 학습 시 "보는 그대로 얻는" 방식이 가능해진다. 따라서 $$\ell_2$$ loss 외에도 perceptual loss를 추가로 고려할 수 있다. Latent 기반 방법은 tokenizer reconstruction 학습 과정에서 perceptual loss를 활용하는 데 유리한 반면, 픽셀 기반 방법은 이러한 이점을 쉽게 활용하지 못했다.

실제로 perceptual loss는 추가된 noise가 특정 threshold 미만, 즉 $$t \le t_\textrm{thr}$$일 때만 적용할 수 있으며, 이 경우 denoising된 이미지가 너무 흐릿해지지 않는다. 본 논문에서는 VGG 기반의 표준 LPIPS loss와 ConvNeXt-V2 기반의 loss를 테스트하였다.

## Experiments
### 1. Prediction Targets of the Network
다음은 2D toy example 결과이다. (네트워크: 7-layer ReLU MLP, 256 hidden units)

<center><img src='{{"/assets/img/pixel-mean-flow/pixel-mean-flow-fig2.webp" | relative_url}}' width="64%"></center>
<br>
다음은 $\textbf{x}$-prediction과 $\textbf{u}$-prediction을 비교한 결과이다.

<center><img src='{{"/assets/img/pixel-mean-flow/pixel-mean-flow-table2.webp" | relative_url}}' width="53%"></center>

### 2. Ablations Studies
다음은 (a) optimizer와 (b) perceptual loss에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/pixel-mean-flow/pixel-mean-flow-fig3.webp" | relative_url}}' width="100%"></center>
<br>
다음은 [pre-conditioner](https://arxiv.org/abs/2206.00364)와 time sampler에 대한 비교 결과이다.

<center><img src='{{"/assets/img/pixel-mean-flow/pixel-mean-flow-table3a.webp" | relative_url}}' width="52%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/pixel-mean-flow/pixel-mean-flow-table3b.webp" | relative_url}}' width="32%"></center>
<br>
다음은 ImageNet에서의 고해상도 생성 결과이다.

<center><img src='{{"/assets/img/pixel-mean-flow/pixel-mean-flow-table4.webp" | relative_url}}' width="52%"></center>
<br>
다음은 scalability를 테스트한 결과이다.

<center><img src='{{"/assets/img/pixel-mean-flow/pixel-mean-flow-table5.webp" | relative_url}}' width="50%"></center>

### 3. System-level Comparisons
다음은 ImageNet 256$\times$256에 대하여 다른 모델들과 비교한 결과이다.

<center><img src='{{"/assets/img/pixel-mean-flow/pixel-mean-flow-table6.webp" | relative_url}}' width="60%"></center>
<br>
다음은 ImageNet 512$\times$512에 대하여 다른 모델들과 비교한 결과이다.

<center><img src='{{"/assets/img/pixel-mean-flow/pixel-mean-flow-table7.webp" | relative_url}}' width="60%"></center>