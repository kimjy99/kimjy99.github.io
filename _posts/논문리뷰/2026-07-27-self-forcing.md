---
title: "[논문리뷰] Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion"
last_modified_at: 2026-07-27
categories:
  - 논문리뷰
tags:
  - Diffusion
  - Video Generation
  - Computer Vision
  - NeurIPS
excerpt: "Self Forcing 논문 리뷰 (NeurIPS 2025 Spotlight)"
use_math: true
classes: wide
---

> NeurIPS 2025 (Spotlight). [[Paper](https://arxiv.org/abs/2506.08009)] [[Page](https://self-forcing.github.io/)] [[Github](https://github.com/guandeh17/Self-Forcing)]  
> Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman  
> Adobe Research | The University of Texas at Austin  
> 9 Jun 2025  

<center><img src='{{"/assets/img/self-forcing/self-forcing-fig1.webp" | relative_url}}' width="90%"></center>

## Introduction
최근 video diffusion model에 autoregressive (AR) 생성 기능을 부여하는 Teacher Forcing (TF)과 [Diffusion Forcing (DF)](https://kimjy99.github.io/논문리뷰/diffusion-forcing)이 등장했다. 시퀀스 모델링에서 잘 알려진 패러다임인 TF는 GT 토큰을 기반으로 다음 토큰을 예측하도록 모델을 학습시킨다. Video diffusion에 적용할 경우, TF는 깨끗한 GT 컨텍스트 프레임을 기반으로 각 프레임을 denoising한다. 반면, DF는 각 프레임에 대해 독립적으로 샘플링된 noise level을 가진 동영상을 사용하여 모델을 학습시키고, noise가 있는 컨텍스트 프레임을 기반으로 각 프레임을 denoising한다. 이를 통해 AR inference 시나리오를 학습 데이터 분포에 포함할 수 있다.

하지만 TF 또는 DF로 학습된 모델은 autoregressive 생성 과정에서 오차가 누적되어 시간이 지남에 따라 동영상 품질이 저하되는 **exposure bias** 문제가 많이 발생한다. 모델은 GT 컨텍스트 데이터만을 사용하여 학습되지만, inference 시에는 불완전한 예측 데이터에 의존해야 하므로 분포 불일치가 발생하고 생성 과정이 진행됨에 따라 오차가 누적된다. 

본 논문에서는 AR 동영상 생성에서 발생하는 exposure bias 문제를 해결하는 **Self Forcing (SF)**을 제안하였다. Self Forcing은 학습 과정에서 AR 생성을 명시적으로 전개함으로써 학습-테스트 데이터 분포의 차이를 해소한다. 즉, 각 프레임을 GT 프레임이 아닌 이전에 생성된 자체 프레임을 기반으로 생성한다. 이를 통해 생성된 전체 동영상 시퀀스에 분포 매칭 loss를 적용하여 전체적인 학습이 가능해진다. Self Forcing은 모델이 자체 예측 오차를 경험하고 이를 통해 학습하도록 함으로써 exposure bias를 효과적으로 완화하고 오차 누적을 줄인다.

Self Forcing은 순차적인 특성으로 인해 병렬 학습이 불가능하여 계산 비용이 많이 드는 것처럼 보일 수 있지만, few-step diffusion backbone과 gradient truncation 전략을 활용함으로써, 다른 병렬 전략보다 훨씬 효율적이며, 동일한 학습 시간 내에 우수한 성능을 달성하였다. 또한, 동영상 extrapolation의 효율성을 향상시키는 **rolling KV cache** 메커니즘을 도입하였다.

## Method
### 1. Autoregressive Diffusion Post-Training via Self-Rollout
Self Forcing의 핵심 아이디어는 inference 시점의 레시피를 따라 학습 중에 autoregressive self-rollout을 통해 동영상을 생성하는 것이다. 구체적으로, 동영상들의 batch을 샘플링한다.

$$
\begin{equation}
\{x_\theta^{1:N}\} \sim p_\theta (x^{1:N}) = \prod_{i=1}^N p_\theta (x^i \vert x^{< i})
\end{equation}
$$

여기서 각 프레임 $x^i$는 과거의 깨끗한 컨텍스트 프레임과 현재 시점의 noise가 포함된 프레임을 포함하여 자체 생성된 출력에 조건부로 반복적인 denoising을 수행하여 생성된다. Inference 중에만 KV caching을 활용하는 대부분의 기존 AR 모델과 달리, Self Forcing은 학습 중에도 KV caching을 활용한다.

<center><img src='{{"/assets/img/self-forcing/self-forcing-fig2.webp" | relative_url}}' width="88%"></center>
<br>
그럼에도 불구하고, 표준 diffusion model을 사용하여 Self Forcing을 구현하는 것은 긴 denoising 체인을 통해 backpropagation을 수행해야 하므로 계산적으로 매우 비효율적이다. 따라서 각 조건부 분포 $$p_\theta (x^i \vert x^{< i})$$를 근사화하기 위해 few-step diffusion model $$G_\theta$$를 사용한다. 

$$\{t_0 = 0, \ldots, t_T = 1000\}$$을 timestep $[0, \ldots, 1000]$의 부분 시퀀스라고 할 때, 각 denoising step $t_j$와 프레임 인덱스 $i$에서 모델은 이전의 깨끗한 프레임 $x^{< i}$를 조건으로 중간 noisy한 프레임 $$x_{t_j}^i$$를 denoising한다. 그런 다음 forward pass $\Psi$를 통해 denoising된 프레임에 더 낮은 noise level의 Gaussian noise를 주입하여 다음 denoising step의 입력으로 사용할 noisy한 프레임 $$x_{t_{j-1}}^i$$을 얻는다. 모델 분포 $$p_\theta (x^i \vert x^{< i})$$는 다음과 같이 implicit하게 정의된다.

$$
\begin{equation}
x^i = f_{\theta, t_1} \circ f_{\theta, t_2} \circ \cdots \circ f_{\theta, t_T} (x_{t_T}^i) \\
\textrm{where} \quad f_{\theta, t_j} (x_{t_j}^i) = \Psi (G_\theta (x_{t_j}^i, t_j, x^{< i}), \epsilon_{t_{j-1}}, t_{j-1}), \quad x_{t_T}^i \sim \mathcal{N}(0,I)
\end{equation}
$$

Few-step 모델이라 하더라도, 전체 AR diffusion process를 통해 backpropagation을 그대로 수행하면 과도한 메모리 소비가 발생한다. 이 문제를 해결하기 위해, 각 프레임의 최종 denoising step에만 backpropagation을 제한하는 gradient truncation 전략을 사용한다. 또한, inference 시처럼 항상 $T$개의 denoising step을 사용하는 대신, 각 학습 iteration에서 각 샘플 시퀀스에 대해 denoising step $s$를 $[1, T]$에서 무작위로 샘플링하고, $s$번째 step의 denoising된 출력을 최종 출력으로 사용한다. 이를 통해 모든 중간 denoising step이 supervision 신호를 받을 수 있다. 추가적으로, 학습 과정에서 KV 캐시 임베딩으로의 gradient flow를 막아 이전 프레임의 gradient와 현재 프레임의 gradient를 분리한다.

### 2. Holistic Distribution Matching Loss
Autoregressive self-rollout은 inference 분포에서 직접 샘플을 생성하므로, 생성된 동영상의 분포 $$p_\theta (x^{1:N})$$를 실제 동영상의 분포 $$p_\textrm{data} (x^{1:N})$$과 일치시키는 전체적인 동영상 수준의 loss를 적용할 수 있다. 사전 학습된 diffusion model을 활용하고 학습 안정성을 향상시키기 위해, 두 분포 모두에 noise를 주입하고 $$p_{\theta,t} (x_t^{1:N})$$와 $$p_{\textrm{data},t} (x_t^{1:N})$$를 일치시킨다. 

본 프레임워크는 다양한 divergence 측정 및 분포 매칭 프레임워크에 일반적으로 적용 가능하며, 본 논문에서는 세 가지 접근 방식을 고려하였다.

1. [Distribution Matching Distillation (DMD)](https://kimjy99.github.io/논문리뷰/dmd): Reverse KL divergence를 최소화.
2. [Score Identity Distillation (SiD)](https://arxiv.org/abs/2404.04057): Fisher divergence를 최소화.
3. [Generative Adversarial Network (GAN)](https://arxiv.org/abs/2501.05441): Jensen–Shannon divergence를 최소화.

세 가지 objective 모두 diffusion model의 timestep distillation 맥락에서 사용되어 왔지만, 단순히 샘플링 속도를 높이는 것이 아니라, 분포 매칭을 통해 exposure bias를 해결하는 것을 목표로 한다. 이러한 차이점 때문에 timestep 감소에만 초점을 맞추고 출력 분포를 직접적으로 정렬하지 않는 다른 distillation 방법들은 적용할 수 없다.

### 3. Long Video Generation with Rolling KV Cache
<center><img src='{{"/assets/img/self-forcing/self-forcing-fig3.webp" | relative_url}}' width="92%"></center>
<br>
표준 video diffusion model에 비해 AR 모델의 주요 장점은 extrapolation 능력으로, 원칙적으로 sliding window inference를 통해 무한히 긴 동영상을 생성할 수 있다. DF로 학습된 bidirectional attention 모델 또한 AR 방식으로 동영상을 생성할 수 있지만, KV caching을 지원하지 않아 각 프레임마다 attention 행렬을 완전히 재계산해야 한다. 이로 인해 $O(T L^2)$의 과도한 계산 복잡도가 발생한다 ($T$는 denoising step 수, $L$은 window 크기).

반면, causal attention을 사용하는 모델은 KV caching을 활용하여 효율성을 향상시킬 수 있다. 그러나 기존 구현에서는 연속적인 sliding window 사이의 겹치는 프레임에 대해 KV 캐시를 다시 계산해야 한다. 이로 인해 dense sliding window를 사용할 경우 $O(L^2 + TL)$의 복잡도가 발생한다. 결과적으로 기존 구현에서는 계산 비용을 줄이기 위해 최소한의 겹침으로 더 큰 stride를 채택했는데, 이는 각 window 시작 부분의 프레임이 상당히 제한된 과거 컨텍스트에 의존하게 되어 시간적 일관성을 저해한다.

저자들은 KV 캐시를 재계산할 필요 없이 무한히 긴 동영상을 생성할 수 있는 **rolling KV cache** 메커니즘을 제안하였다. 가장 최근 $L$개 프레임의 토큰에 대한 KV 임베딩을 저장하는 고정 크기의 KV 캐시를 유지한다. 새로운 프레임을 생성할 때, 먼저 KV 캐시가 가득 찼는지 확인한다. 가득 찼다면, 새 entry를 추가하기 전에 가장 오래된 KV 캐시 entry 항목을 제거한다. 이를 통해 각 새 프레임을 생성할 때 충분한 컨텍스트 길이를 유지하면서 $O(TL)$의 시간 복잡도로 무한히 프레임을 생성할 수 있다. 

하지만 이 메커니즘을 단순하게 구현하면 분포 불일치로 인해 심각한 깜빡임 현상이 발생한다. 특히 첫 번째 latent 프레임은 다른 프레임과 통계적 특성이 다르다. 즉, 시간 압축을 수행하지 않고 첫 번째 이미지만 인코딩한다. 모델은 학습 과정에서 항상 첫 번째 프레임을 이미지 latent로 인식해 왔기 때문에, rolling KV cache 시나리오에서 이 이미지 latent가 더 이상 보이지 않을 때 일반화에 실패한다. 따라서 저자들은 학습 과정에서 attention window를 제한하여 모델이 마지막 청크의 denoising 시 첫 번째 청크에 attention하지 못하도록 함으로써 긴 동영상 생성 시 발생하는 조건을 시뮬레이션하였다.

## Experiments
- Base model: Wan2.1-T2V-1.3B (5초, 16 FPS, 832$\times$480)

### 1. Comparison with existing baselines
다음은 [VBench](https://arxiv.org/abs/2311.17982)로 비교한 결과이다.

<center><img src='{{"/assets/img/self-forcing/self-forcing-fig5.webp" | relative_url}}' width="100%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/self-forcing/self-forcing-table1.webp" | relative_url}}' width="80%"></center>
<br>
다음은 user study 결과이다.

<center><img src='{{"/assets/img/self-forcing/self-forcing-fig4.webp" | relative_url}}' width="45%"></center>

### 2. Ablation Studies
다음은 ablation study 결과이다.

<center><img src='{{"/assets/img/self-forcing/self-forcing-table2.webp" | relative_url}}' width="82%"></center>

### 3. Training efficiency
다음은 학습 효율성을 비교한 결과이다.

<center><img src='{{"/assets/img/self-forcing/self-forcing-fig6.webp" | relative_url}}' width="95%"></center>