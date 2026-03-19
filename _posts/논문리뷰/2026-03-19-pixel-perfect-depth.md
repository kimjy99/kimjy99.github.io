---
title: "[논문리뷰] Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers"
last_modified_at: 2026-03-19
categories:
  - 논문리뷰
tags:
  - DiT
  - Diffusion
  - Monocular Depth Estimation
  - Computer Vision
  - NeurIPS
excerpt: "Pixel-Perfect Depth 논문 리뷰 (NeurIPS 2025)"
use_math: true
classes: wide
---

> NeurIPS 2025. [[Paper](https://arxiv.org/abs/2510.07316)] [[Page](https://pixel-perfect-depth.github.io/)] [[Github](https://github.com/gangweix/pixel-perfect-depth)]  
> Gangwei Xu, Haotong Lin, Hongcheng Luo, Xianqi Wang, Jingfeng Yao, Lianghui Zhu, Yuechuan Pu, Cheng Chi, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Sida Peng, Xin Yang  
> Huazhong University of Science and Technology | Xiaomi EV | Zhejiang University  
> 8 Oct 2025  

<center><img src='{{"/assets/img/pixel-perfect-depth/pixel-perfect-depth-fig1.webp" | relative_url}}' width="100%"></center>

## Introduction
최근 monocular depth estimation (MDE) 모델은 대부분의 zero-shot 시나리오 또는 영역에서 고품질 결과를 얻지만, object 경계 주변의 flying pixel이 생기고 포인트 클라우드로 변환될 때 미세한 디테일이 제대로 표현되지 않는 문제가 있다.

본 논문에서는 픽셀 공간 DiT를 사용하여 flying pixel이 없는 고품질의 monocular depth estimation을 위한 프레임워크인 **Pixel-Perfect Depth**를 제시하였다. 픽셀 공간에서의 고해상도 생성은 글로벌한 이미지 구조를 인식하고 모델링하는 데 주요 어려움이 있다. 본 논문은 이 문제를 해결하기 위해 고수준 semantic 표현을 diffusion process에 통합하여 모델의 글로벌 구조 및 semantic 일관성 보존 능력을 향상시키는 **Semantics-Prompted Diffusion Transformers (SP-DiT)**를 제안하였다. SP-DiT를 적용한 본 모델은 고해상도 픽셀 공간에서 세밀한 시각적 디테일을 생성하는 동시에 글로벌 semantic 일관성을 더욱 효과적으로 유지할 수 있다. 그러나 vision foundation model에서 얻은 semantic 표현이 DiT의 내부 표현과 잘 일치하지 않아 학습 불안정성과 수렴 문제가 발생한다. 이를 해결하기 위해 semantic 표현에 대한 간단하면서도 효과적인 정규화 기법을 도입하여 안정적인 학습을 보장하고 원하는 해로의 수렴을 촉진한다.

또한, 본 논문에서는 DiT를 위한 효율적인 아키텍처인 **Cascade DiT (Cas-DiT)** 디자인을 도입하였다. DiT에서 초기 block은 주로 글로벌 또는 저주파 구조를 포착하고 생성하는 역할을 하는 반면, 나중 block은 고주파 디테일을 생성하는 데 집중한다. 이를 바탕으로 Cas-DiT는 점진적인 patch size를 사용하는 전략을 채택하였다. 초기 DiT block에서는 토큰 수를 줄이고 글로벌 이미지 구조 모델링을 용이하게 하기 위해 더 큰 patch size를 사용하고, 나중 DiT block에서는 토큰 수를 늘려, 즉 patch size를 줄여 모델이 세밀한 공간 디테일 생성에 집중할 수 있도록 한다. 이러한 세밀한 구조로의 계단식 디자인은 계산 비용을 크게 줄이고 효율성을 향상시킬 뿐만 아니라 정확도 또한 크게 향상시킨다.

## Method
### 1. Generative Formulation
본 논문에서는 깊이 추정 프레임워크의 생성 핵심으로 Flow Matching을 채택하였다. Flow Matching은 1차 상미분 방정식(ODE)을 통해 Gaussian noise에서 데이터 샘플로의 연속적인 변환을 학습한다. 본 논문에서는 Gaussian noise에서 깊이 샘플로의 변환을 모델링한다.

구체적으로, 깨끗한 깊이 샘플 $x_0 \sim \mathcal{D}$와 Gaussian noise $x_1 \sim \mathcal{N}(0,1)$이 주어졌을 때, 시간 $t \in [0, 1]에$서의 보간된 샘플을 다음과 같이 정의한다.

$$
\begin{equation}
\textbf{x}_t = t \cdot \textbf{x}_1 + (1 - t) \cdot \textbf{x}_0
\end{equation}
$$

이는 velocity field를 정의한다.

$$
\begin{equation}
\textbf{v}_t = \frac{\textrm{d}\textbf{x}_t}{\textrm{d}t} = \textbf{x}_1 - \textbf{x}_0
\end{equation}
$$

이는 깨끗한 데이터에서 noise로의 방향을 나타낸다. 모델 $$\textbf{v}_\theta (\textbf{x}_t, t, \textbf{c})$$는 현재의 noisy한 샘플 $$\textbf{x}_t$$, timestep $t$, 그리고 입력 이미지 $\textbf{c}$를 기반으로 velocity field를 예측하도록 학습되었다. 학습 loss는 예측된 속도와 실제 속도 사이의 MSE이다.

$$
\begin{equation}
\mathcal{L}_\textrm{velocity} = \mathbb{E}_{\textbf{x}_0, \textbf{x}_1, t} \left[ \| \textbf{v}_\theta (\textbf{x}_t, t, \textbf{c}) - \textbf{v}_t \|^2 \right]
\end{equation}
$$

Inference 단계에서는 noise $$\textbf{x}_1$$에서 시작하여 시간 간격 $[0, 1]$을 timestep $t_i$로 discretize하고 다음과 같이 깊이 샘플을 반복적으로 업데이트하여 ODE를 푼다.

$$
\begin{equation}
\textbf{x}_{t_{i-1}} = \textbf{x}_{t_i} + \textbf{v}_\theta (\textbf{x}_t, t, \textbf{c}) (\textbf{t}_{i-1} - \textbf{t}_i)
\end{equation}
$$

여기서 $t_i$는 1에서 0으로 감소하며, 초기 noise $$\textbf{x}_1$$은 점진적으로 깊이 샘플 $$\textbf{x}_0$$으로 변환된다.

### 2. Semantics-Prompted Diffusion Transformers
<center><img src='{{"/assets/img/pixel-perfect-depth/pixel-perfect-depth-fig3.webp" | relative_url}}' width="80%"></center>
<br>
Semantics-Prompted DiT (SP-DiT)는 DiT를 기반으로 한다. Depth Anything v2나 Marigold와 같은 기존의 깊이 추정 모델과는 달리, SP-DiT 아키텍처는 convolutional layer 없이 순수하게 transformer 기반으로 설계되었다. 고수준의 semantic 표현을 통합함으로써, SP-DiT는 DiT의 단순성과 scalability를 유지하면서도 글로벌한 semantic 일관성을 보존하고 세밀한 시각적 디테일을 향상시킬 수 있다.

구체적으로, 보간된 샘플 $$\textbf{x}_t$$와 이에 대응하는 이미지 $\textbf{c}$가 주어졌을 때, 먼저 이들을 하나의 입력으로 concat한다. 

$$
\begin{equation}
\textbf{a}_t = \textbf{x}_t \oplus \textbf{c}
\end{equation}
$$

여기서 이미지 $\textbf{c}$는 조건으로 작용한다. 그런 다음, $$\textbf{a}_t$$를 DiT에 직접 입력한다. DiT의 첫 번째 layer는 patchify 연산으로, 공간 입력 $$\textbf{a}_t$$를 각각 크기가 $D$인 $T$개의 토큰(패치)으로 이루어진 1차원 시퀀스로 변환한다. 이는 입력 $$\textbf{a}_t$$에서 크기가 $p \times p$인 각 패치를 선형적으로 임베딩하는 방식으로 이루어진다. 이후, 입력 토큰들은 DiT block이라고 불리는 일련의 Transformer block들을 거친다. 마지막 DiT block을 통과한 후, 각 토큰은 $p \times p$ 텐서로 linear projection되고, 다시 원래의 공간 해상도로 reshape되어 채널 차원이 1인 예측 속도 $$\textbf{v}_t = \textbf{x}_1 - \textbf{x}_0$$을 얻는다.

Pixel-space에서 직접 diffusion을 수행하면 수렴이 제대로 되지 않고 깊이 예측이 매우 부정확해진다. 모델은 이미지의 글로벌 구조와 미세한 디테일을 모두 모델링하는 데 어려움을 겪는다. 이러한 문제를 해결하기 위해 다음과 같이 vision foundation model $f$를 사용하여 입력 이미지 $\textbf{c}$에서 고수준 semantic 표현 $\textbf{e}$를 guidance로 추출한다.

$$
\begin{equation}
\textbf{e} = f (\textbf{c}) \in \mathbb{R}^{T^\prime \times D^\prime}
\end{equation}
$$

($T^\prime$와 $D^\prime$은 각각 토큰 수와 임베딩 차원)

이러한 고수준 semantic 표현은 DiT 모델에 통합되어, 세밀한 시각적 디테일을 향상시키면서 전반적인 semantic 일관성을 더욱 효과적으로 유지할 수 있도록 한다. 그러나 얻어진 semantic 표현 $\textbf{e}$의 크기가 DiT 모델의 토큰 크기와 크게 다르며, 이는 모델 학습의 안정성과 성능 모두에 영향을 미친다. 이를 해결하기 위해 다음과 같이 L2 norm을 사용하여 feature 차원을 따라 semantic 표현 $\textbf{e}$를 정규화한다.

$$
\begin{equation}
\hat{\textbf{e}} = \frac{\textbf{e}}{\| \textbf{e} \|_2}
\end{equation}
$$

이후, 정규화된 semantic 표현은 MLP layer $h_\phi$를 통해 DiT 모델의 토큰 $\textbf{z}$에 통합된다.

$$
\begin{equation}
\textbf{z}^\prime = h_\phi (\textbf{z} \oplus \mathcal{B}(\hat{\textbf{e}}))
\end{equation}
$$

($$\mathcal{B}(\cdot)$$는 semantic 표현 $\hat{\textbf{e}}$의 공간 해상도를 DiT 토큰의 공간 해상도와 일치시키는 bilinear interpolation 연산자)

결과적으로 $$\textbf{z}^\prime$$은 semantic 정보로 향상된 DiT 토큰이 된다. 융합 후, 후속 DiT block들은 semantic 정보를 통해 글로벌 semantic 일관성을 효과적으로 유지하면서 고해상도 픽셀 공간에서 세밀한 시각적 디테일을 향상시킨다. 이러한 후속 DiT block들을 SP-DiT라고 부른다.

### 3. Cascade DiT Design
SP-DiT는 정확도 성능을 크게 향상시키지만, 픽셀 공간에서 직접 diffusion을 수행하는 것은 여전히 ​​계산 비용이 많이 든다. 이러한 문제를 해결하기 위해, 본 논문에서는 모델의 계산 부담을 줄이는 새로운 Cascaded DiT 디자인을 제안하였다. DiT 아키텍처에서 초기 block들은 주로 글로벌한 이미지 구조와 저주파 정보를 포착하는 역할을 하고, 나중 block들은 세밀한 고주파 디테일을 모델링하는 데 집중한다.

이 과정의 효율성과 효과를 최적화하기 위해 초기 DiT block에서는 큰 patch size를 사용한다. 이러한 디자인은 처리해야 하는 토큰 수를 크게 줄여 계산 비용을 낮춘다. 또한, 모델이 글로벌한 이미지 구조와 저주파 정보를 학습하고 모델링하는 데 우선순위를 두도록 유도하여 입력 이미지에서 추출한 고수준 semantic 표현과 더 잘 부합하도록 한다. 후반부 DiT block에서는 토큰 수를 늘리는데, 이는 더 작은 patch size를 사용하는 것과 같다. 이를 통해 모델은 미세한 공간적 디테일에 더 집중할 수 있다. 결과적으로 이 coarse-to-fine 디자인은 시각적 인식의 계층적 특성을 반영하여 깊이 추정의 효율성과 정확도를 모두 향상시킨다.

구체적으로, 총 $N$개의 DiT block으로 구성된 diffusion model의 경우, 처음 $N/2$개의 block은 더 큰 patch size를 갖는 coarse stage를 구성하고, 나머지 N/2개의 block(즉, SP-DiT)은 더 작은 patch size를 사용하는 fine stage를 구성한다.

### 4. Implementation Details
##### 모델 아키텍처 디테일
총 $N = 24$개의 DiT block을 사용하며, 각 block은 hidden dimension $D = 1024$에서 동작한다. 처음 12개의 block은 patch size가 16인 표준 DiT block으로, 입력 크기가 H$\times$W일 때 $(H/16) \times (W/16)$개의 토큰에 해당한다. 12번째 block 이후에는 MLP layer를 사용하여 hidden dimension을 4배로 확장하고, $(H/8) \times (W/8)$ 토큰으로 reshape한다. 나머지 12개의 SP-DiT block은 이러한 $(H/8) \times (W/8)$ 토큰을 추가로 처리한다. 마지막으로, MLP layer와 reshape 연산을 통해 처리된 토큰을 $H \times W$ depth map으로 변환한다.

##### 깊이 정규화
실제 깊이 값은 diffusion model에서 예상하는 scale에 맞게 정규화된다. 실내 및 실외 장면 모두를 커버하기 위해, 정규화 전에 깊이 값을 로그 스케일로 변환한다.

$$
\begin{equation}
\tilde{\textbf{d}} = \log (\textbf{d} + 1)
\end{equation}
$$

그런 다음 로그 스케일로 변환된 깊이 $\tilde{\textbf{d}}$를 다음과 같이 정규화한다.

$$
\begin{equation}
\hat{\textbf{d}} = \frac{\tilde{\textbf{d}} - d_\textrm{min}}{d_\textrm{max} - d_\textrm{min}} - 0.5
\end{equation}
$$

($$d_\textrm{min}$$과 $$d_\textrm{max}$$는 각 depth map의 2% 및 98% 깊이 백분위수)

## Experiments
- 학습 디테일
  - 해상도: 512$\times$512, 1024$\times$768
  - optimizer: AdamW
  - 총 batch size: 32 (GPU당 4)
  - learning rate: $1 \times 10^{-4}$
- 데이터셋
  - 512$\times$512: Hypersim
  - 1024$\times$768: Hypersim, UrbanSyn, UnrealStereo4K, VKITTI, TartanAir

### 1. Ablations and Analysis
다음은 SP-DiT와 Cas-DiT에 대한 ablation study 결과이다.

<center><img src='{{"/assets/img/pixel-perfect-depth/pixel-perfect-depth-fig6.webp" | relative_url}}' width="100%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/pixel-perfect-depth/pixel-perfect-depth-table2.webp" | relative_url}}' width="74%"></center>
<br>
다음은 vision foundation model에 대한 ablation study 결과이다.

<center><img src='{{"/assets/img/pixel-perfect-depth/pixel-perfect-depth-table3.webp" | relative_url}}' width="75%"></center>

### 2. Zero-Shot Relative Depth estimation
다음은 zero-shot relative depth 추정 결과이다.

<center><img src='{{"/assets/img/pixel-perfect-depth/pixel-perfect-depth-table1.webp" | relative_url}}' width="76%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/pixel-perfect-depth/pixel-perfect-depth-fig4.webp" | relative_url}}' width="85%"></center>

### 3. Edge-Aware Point Cloud Evaluation
다음은 포인트 클라우드를 비교한 결과이다.

<center><img src='{{"/assets/img/pixel-perfect-depth/pixel-perfect-depth-table4.webp" | relative_url}}' width="70%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/pixel-perfect-depth/pixel-perfect-depth-fig5.webp" | relative_url}}' width="100%"></center>