---
title: "[논문리뷰] Deterministic Differentiable Structured Pruning for Large Language Models"
last_modified_at: 2026-08-24
categories:
  - 논문리뷰
tags:
  - NLP
  - LLM
  - ICML
excerpt: "Deterministic Differentiable Pruning (DDP) 논문 리뷰 (ICML 2026)"
use_math: true
classes: wide
---

> ICML 2026. [[Paper](https://arxiv.org/abs/2603.08065)] [[Github](https://github.com/yellowtree123/Deterministic-Differentiable-Pruning)]  
> Weiyu Huang, Pengle Zhang, Xiaolu Zhang, Jun Zhou, Jun Zhu, Jianfei Chen  
> Carnegie Mellon University | AMD | William & Mary | MBZUAI  
> 9 Mar 2026  

## Introduction
Structured pruning은 모델 크기와 추론 비용을 줄이기 위해 전체 아키텍처 구성 요소를 제거함으로써 이러한 비용을 절감하는 방법이다. 최신 LLM에 사용되는 대부분의 structured pruning 방법은 휴리스틱 중요도 점수를 사용하여 구성요소를 선택하는 효율적인 일회성 접근 방식에 의존한다. 이러한 휴리스틱은 빠르다는 장점이 있지만, 불안정할 수 있으며, sparsity가 심한 경우 품질 저하가 크게 발생할 수 있다. 반면, end-to-end 가중치 업데이트를 통해 sparsity를 학습하는 방식은 LLM 규모에서는 비용 부담이 크다.

Mask-only 최적화가 휴리스틱 pruning과 전체 가중치 튜닝 사이의 실용적인 중간 지점을 제공한다. 여기서 마스크는 각 아키텍처 구성 요소를 유지할지 또는 제거할지를 결정하는 학습 가능한 gating 변수 집합이다. 본 논문에서는 모든 사전 학습된 가중치를 고정하고 마스크만 최적화하였다. 이렇게 하면 탐색 공간이 훨씬 작다. 685B DeepSeek-R1의 경우 마스크 수는 수천만 개에 불과하여 (심지어 LoRA 모듈보다 작음) 적은 토큰 예산 내에서 수렴하는 scalable한 gradient 기반 최적화가 가능하다.

기존의 mask-only 방식은 일반적으로 가중치 업데이트와 함께 마스크를 학습하므로 학습 비용이 증가하고 안정적인 복구를 위해 대규모 사전 학습 데이터가 필요한 경우가 많다. 또한, 일반적으로 stochastic hard-concrete relaxation에 의존한다. 이는 마스크를 제한된 binary 범위로 제한하고 샘플링 노이즈를 도입하여 수렴 속도를 늦추고 deterministic한 마스크가 필요한 경우에 학습-테스트 불일치를 초래할 수 있다.

이러한 문제들을 해결하기 위해, 본 논문에서는 gradient 기반 최적화를 통해 구조화된 sparsity 패턴을 학습하는 mask-only pruning 프레임워크인 **Deterministic Differentiable Pruning (DDP)**를 제안하였다. Pruning을 $$\ell_0$$ 정규화 최적화 문제로 구성하고, augmented Lagrangian method (ALM)를 사용하여 sparsity 제약 조건을 적용한다. $$\ell_0$$ norm의 미분 불가능성을 처리하기 위해, DDP는 학습 과정에서 어닐링되는 deterministic smooth surrogate를 도입하여 샘플링 노이즈를 제거하고 확률적 마스크로 인한 학습-테스트 불일치를 방지하였다. 또한, forward pass에 사용되는 마스크 값과 정규화에 사용되는 마스크 값을 분리하여, 유효 마스크 값 범위를 확장하고 성능을 향상시켰다. 추가로 마스크 양극화를 유도하는 binarization loss를 사용하여 더 빠르고 안정적인 수렴을 가능하게 하였다.

본 논문에서는 DDP를 dense 모델과 MoE 모델 모두에서 검증하고, 수백억 개의 파라미터를 가진 모델까지 scaling했다. DDP는 기존 SOTA 방법보다 일관적으로 우수한 품질-효율성 trade-off를 제공한다.

## Preliminary
### 1. Unified Masking Formulation
Transformer 기반 LLM은 multi-head attention 및 MLP와 같은 반복적인 서브모듈로 구성되며, 출력은 $K$개의 독립적인 구성 요소의 합으로 표현될 수 있다. 본 논문은 각 구성 요소에 마스크를 적용하여 structured pruning을 모델링하였다. 입력 $\textbf{X}$가 주어졌을 때, 모듈 출력은 다음과 같다.

$$
\begin{equation}
\textbf{y} = \sum_{k=1}^K m_k \textbf{f}_k (\textbf{X}), \quad m_k \in \mathbb{R}
\end{equation}
$$

($$f_k (\cdot)$$는 $k$번째 구성 요소. $m_k$는 해당 gate. $m_k = 0$은 제거, $m_k \ne 0$은 유지)

Dense LLM의 경우, 이를 attention head와 MLP block 모두에 적용한다. 각 layer마다 attention head에 대한 마스크 $$\textbf{m}^\textrm{attn} \in \mathbb{R}^H$$와 MLP block에 대한 마스크 $$\textbf{m}^\textrm{mlp} \in \mathbb{R}^C$$가 사용된다.

### 2. Mask Optimization by $$\ell_0$$ Regularization
본 논문의 목표는 주어진 예산을 만족하면서 언어 모델링 loss를 최소화하는 마스크 모음 $$M = \{\textbf{m}^\textrm{attn}, \textbf{m}^\textrm{mlp}\}$$을 최적화하는 것이다. 자연스러운 시작점은 $$\ell_0$$ 정규화를 사용한 학습이다.

$$
\begin{equation}
\min_{\textbf{m}} \mathcal{L}_\textrm{ce} (\theta, \mathcal{m}) + \lambda \| \textbf{m} \|_0
\end{equation}
$$

($$\mathcal{L}_\textrm{ce}$$는 cross-entropy loss)

실제로는 정규화 강도 $\lambda$를 조정하는 것보다 명시적인 sparsity level이 필요한 경우가 많다. 따라서 제약 조건이 있는 최적화 문제로 다시 표현할 수 있다.

$$
\begin{equation}
\min_{\textbf{m} \in \mathcal{S}} \mathcal{L}_\textrm{ce} (\theta, \textbf{m})
\end{equation}
$$

($\mathcal{S}$는 목표 제약 조건 집합)

특히, $\textbf{m}$에 대해 목표 유지 비율 $\rho$를 적용한다.

$$
\begin{equation}
\bar{m} = \frac{1}{K} \| \textbf{m} \|_0 = \frac{1}{K} \sum_{k=1}^K \| m_k \|_0 = \rho
\end{equation}
$$

현재의 최적화 방법은 경사 하강법에 기반하고 있으며 제약 조건이 없는 매끄러운 loss에 맞춰져 있기 때문에 위 식을 직접 푸는 것은 어렵다. 일반적인 해결책은 augmented Lagrangian method (ALM)을 통해 제약 조건이 없는 loss로 변환하는 것이다. 구체적으로, 각 $\textbf{m}$에 대해 다음과 같은 형태의 sparsity 페널티를 추가한다.

$$
\begin{equation}
\mathcal{L}_\textrm{sparsity} (\| \textbf{m} \|_0) = \lambda_1 (\bar{m} - \rho) + \lambda_2 (\bar{m} - \rho)^2
\end{equation}
$$

##### Hard-Concrete Relaxation
ALM은 $$\ell_0$$ 정규화 항이 미분 불가능하다는 근본적인 어려움이 여전히 ​​남아 있다. 기존 방법들에서는 hard-concrete relaxation 기법을 채택하여 $\textbf{m}$을 $$\ell_0$$ norm을 모델링하는 0과 1 사이의 확률 변수로 재정의했다. Forward pass에서 마스크 $\textbf{m}$은 hard-concrete 매핑 $$\Phi (\textbf{z}, \textbf{u})$$를 사용하여 생성되며, 보조 변수 $\textbf{u}$를 통해 랜덤성이 주입된다.

$$
\begin{aligned}
& \textbf{u} \sim \mathcal{U}(0, 1), \quad \textbf{v} = \sigma (\log \textbf{u} - \log (1 - \textbf{u}) + \textbf{z}), \\
& \bar{\textbf{v}} = \textbf{v} (r - l) + l, \quad \textbf{m} = \textrm{Clamp} (\bar{\textbf{v}}, 0, 1)
\end{aligned}
$$

($\sigma (\cdot)$는 sigmoid function, $l < 0 < 1 < r$은 stretching 파라미터, $\textbf{z}$는 최적화될 파라미터)

이 relaxation은 다음과 같은 $$\ell_0$$ norm 기대값에 대한 closed-form 근사치를 제공한다.

$$
\begin{equation}
\mathbb{E}[\| \textbf{m} \|_0] = \sum_{k=0}^K \mathbb{P}(m_k > 0) = \sum_{k=1}^K \sigma (z_k - \log (-l/r))
\end{equation}
$$

### 3. Drawbacks of Hard-Concrete Relaxation
Hard-concrete 방법에서 사용되는 loss는 다음과 같다.

$$
\begin{equation}
\min_\textbf{z} \mathbb{E}_{\textbf{u} \sim \mathcal{U}(0, 1)} \left[ \mathcal{L}_\textrm{ce} (\theta, \mathcal{m}) + \mathcal{L}_\textrm{sparsity} (\| \textbf{m} \|_0) \right]
\end{equation}
$$

이 식에는 두 가지 주요 단점이 있다.

1. **학습-테스트 불일치**. 학습 중에는 마스크 $\textbf{m}$이 샘플링되지만, 테스트 시에는 deterministic한 마스크가 필요하다. 확률 변수 $\textbf{m}$을 discrete한 값으로 변환하면 학습-테스트 불일치가 발생하여 성능 저하로 이어질 수 있다. 또한 랜덤성 때문에 수렴 속도를 늦춘다.
2. **제한된 마스크 표현력**. $\textbf{m}$은 $$\mathcal{L}_\textrm{ce}$$를 계산할 때 자연스럽게 실수값을 갖는다. 그러나 hard-concrete 매핑은 $\textbf{m}$을 거의 0과 1 사이로 제한하여 탐색 공간을 축소하고 고품질의 sparsity 패턴 발견을 방해할 수 있다.

## Method
<center><img src='{{"/assets/img/deterministic-differentiable-pruning/deterministic-differentiable-pruning-fig1.webp" | relative_url}}' width="100%"></center>

### 1. Deterministic Differentiable Pruning
먼저, 실수값 파라미터 $\textbf{z}$가 주어졌을 때, forward pass에서 hard-concrete 샘플링을 deterministic한 ReLU gate로 대체한다.

$$
\begin{equation}
\textbf{m} = \textrm{ReLU}(\textbf{z})
\end{equation}
$$

이는 마스크 공간을 $m_k \in [0, \infty)$까지 확장하여 구성 요소 기여도의 continuous scaling을 가능하게 하는 동시에 부호 반전 및 원치 않는 상쇄를 유발할 수 있는 음수 마스크 값을 방지한다.

또한, 확률적 샘플링을 도입하지 않고 $$\ell_0$$ norm의 미분 불가능성을 해결하기 위해, ALM sparsity 항을 계산할 때 사용되는 retention score $\textbf{s} \in [0, 1]$를 사용한다. $\textbf{s}$는 $\textbf{z}$로 부터 deterministic하게 $$\textbf{s} = \phi (\textbf{z}; \mu_t)$$로 매핑된다.

$$
\begin{equation}
\textbf{v} = \sigma \left( (\textbf{z} - \mu_t) \frac{C_0}{\mu_t} \right), \quad \bar{\textbf{v}} = \textbf{v} (r - l) + l, \\
\textbf{s} = \textrm{Clamp} (\bar{\textbf{v}}, 0, 1)
\end{equation}
$$

($\mu_t$는 sharpness 파라미터, $l = -0.1$, $r = 1.1$, $C_0 \approx 2.4$)

이렇게 얻은 $\textbf{s}$ 값들은 sparsity 항에 사용되어 목표 유지 비율 $\rho$를 강제한다.

$$
\begin{equation}
\mathcal{L}_\textrm{sparsity} (\textbf{s}) = \lambda_1 (\bar{s} - \rho) + \lambda_2 (\bar{s} - \rho)^2 \\
\textrm{where} \quad \bar{s} = \frac{1}{K} \sum_k s_k
\end{equation}
$$

모든 $k$에 대해 $z_k = 1$로 초기화한다. $T$를 전체 학습 step 수라고 하면, $$\mu_t$$를 다음과 같이 어닐링한다.

$$
\begin{equation}
\mu_t = \mu_0 - (\mu_0 - \mu_T) \sqrt{\frac{t}{T}}
\end{equation}
$$

($$\mu_0 = 0.5$$, $$\mu_T = 0.05$$)

<center><img src='{{"/assets/img/deterministic-differentiable-pruning/deterministic-differentiable-pruning-fig2.webp" | relative_url}}' width="50%"></center>
<br>
매핑은 시간이 지남에 따라 점점 더 날카로워지므로, $$\mu_t \rightarrow 0$$일 때 정확한 $$\ell_0$$로 동작한다. Gradient 보존을 위해 clamping과 ReLU에 straight-through estimator (STE)를 사용한다.

Sparsity 항은 평균 유지 비율을 강제하지만, 개별 유지 점수가 0 또는 1에 가까워지도록 그 자체로는 유도하지 않는다. 수렴 속도를 높이기 위해 $$\{s_k\}_{k=1}^K$$에 추가적인 binarization 항을 도입한다.

$$
\begin{equation}
\mathcal{L}_\textrm{bin} (\textbf{s}) = \lambda_3 \frac{1}{K} \sum_{k=1}^K s_k (1 - s_k)
\end{equation}
$$

이 항은 모호한 구성 요소가 조기에 결정을 내리도록 유도하여 최적화를 안정화한다. 실제로는, 마스크 파라미터와 함께 gradient ascent로 $$\lambda$$ 계수들을 업데이트한다. 학습 후 마스크 값이 0인 구성 요소를 제거하고 나머지 0이 아닌 마스크를 모델 파라미터에 포함시킨다.

최종 loss는 다음과 같다.

$$
\begin{equation}
\min_\textbf{z} \mathcal{L}_\textrm{ce} (\theta, \textbf{m}) + \mathcal{L}_\textrm{sparsity} (\textbf{s}) + \mathcal{L}_\textrm{bin} (\textbf{s})
\end{equation}
$$

### 2. Extensions
##### Distillation
사전 학습된 가중치를 고정하고 마스크만 최적화하므로, dense 모델은 teacher 역할을 한다. 따라서 추가적인 optimizer 메모리 없이 두 번의 forward pass로 distillation을 할 수 있다. Teacher 모델과 student 모델 간의 KL divergence는 다음과 같이 정의한다.

$$
\begin{equation}
\mathcal{L}_\textrm{kl} (\theta, \textbf{m}) = \sum_i D_\textrm{kl} (P_t (\textbf{X}, i) \, \| \, P_s (\textbf{X}, i))
\end{equation}
$$

($$P_t (\textbf{X}, i)$$와 $$P_s (\textbf{X}, i)$$는 각각 위치 $i$에서의 teacher와 student의 다음 토큰 분포)

##### Mixture-of-Experts (MoE)
MoE 모델의 경우, 대부분의 파라미터가 expert MLP에 존재하므로 expert에만 pruning을 적용하고 attention block은 변경하지 않는다. 각 expert 내에서 동일한 채널별 분해를 사용하고 router score $$\pi_e$$로 expert 출력에 가중치를 부여한다.

$$
\begin{equation}
\textrm{MoE} (\textbf{X}) = \sum_{e=1}^E \pi_e (\textbf{X}) \sum_{j=1}^C m_{e,j} \textbf{f}_{e,j}^\textrm{mlp} (\textbf{X})
\end{equation}
$$

($E$는 expert 수, $C$는 expert당 중간 채널 차원)

따라서 MoE layer 1개에 대한 마스크 파라미터는 $\textbf{m}^\textrm{moe} \in \mathbb{R}^{E \times C}$이다.

##### Fine-grained Sparsity Control
DDP는 그룹 수준에서 sparsity loss를 적용하여 다양한 sparsity 수준을 지원한다. 마스크를 그룹 $\mathcal{G}$로 분할하고 (ex. layer별 또는 expert별) 그룹별 유지 비율 예산을 적용할 수 있다.

$$
\begin{equation}
\mathcal{L}_\textrm{sparsity} (\textbf{s}) = \frac{1}{\vert \mathcal{G} \vert} \sum_{g \in \mathcal{G}} \left[ \lambda_1 (\bar{s}_g - \rho) + \lambda_2 (\bar{s}_g - \rho)^2 \right] \\
\textrm{where} \quad \bar{s}_g = \frac{1}{\vert g \vert} \sum_{k \in g} s_k
\end{equation}
$$

이를 통해 layer 전체 또는 각 MoE expert 내에서 균일한 sparsity를 확보할 수 있으며, 보다 규칙적이고 하드웨어 친화적인 패턴을 생성하여 속도 향상을 가져오는 경우가 많다.

## Expertiments
- sparsity ratio $\alpha = 1 - rho$
- 데이터셋: FineWeb-Edu 30M 토큰
- 구현 디테일
  - Dense base model: LLaMA, Qwen3
  - MoE base model: DeepSeekMoE-16B, Qwen3-30B-A3B
  - GPU: NVIDIA H20 GPU 4개

### 1. Main Results
다음은 dense 모델에 대한 비교 결과이다.

<center><img src='{{"/assets/img/deterministic-differentiable-pruning/deterministic-differentiable-pruning-table2.webp" | relative_url}}' width="100%"></center>
<br>
다음은 MoE 모델에 대한 비교 결과이다.

<center><img src='{{"/assets/img/deterministic-differentiable-pruning/deterministic-differentiable-pruning-table3.webp" | relative_url}}' width="100%"></center>

### 2. Ablation Study
다음은 ablation 결과이다. (HC: stochastic hard-concrete; Det. HC: deterministic hard-concrete; +EM: expanded-mask parameterization)

<center><img src='{{"/assets/img/deterministic-differentiable-pruning/deterministic-differentiable-pruning-table4.webp" | relative_url}}' width="47%"></center>
<br>
다음은 그룹 수준의 sparsity loss를 사용한 결과이다. (global sparsity loss와 비교)

<center><img src='{{"/assets/img/deterministic-differentiable-pruning/deterministic-differentiable-pruning-table5.webp" | relative_url}}' width="50%"></center>
<br>
다음은 학습에 사용한 토큰 수에 따른 성능을 비교한 결과이다.

<center><img src='{{"/assets/img/deterministic-differentiable-pruning/deterministic-differentiable-pruning-fig3.webp" | relative_url}}' width="100%"></center>

### 3. Mask-Only Optimization vs. LoRA Recovery
다음은 일반적인 2단계 전략 (training-free pruning 후 LoRA fine-tuning)과 비교한 결과이다.

<center><img src='{{"/assets/img/deterministic-differentiable-pruning/deterministic-differentiable-pruning-table7.webp" | relative_url}}' width="51%"></center>

### 4. Computational Cost
다음은 학습 비용을 비교한 결과이다. (LLaMA2-7B)

<center><img src='{{"/assets/img/deterministic-differentiable-pruning/deterministic-differentiable-pruning-table9.webp" | relative_url}}' width="47%"></center>
<br>
다음은 (왼쪽) dense 모델과 (오른쪽) MoE 모델에 대한 속도 개선 결과이다.

<div style="display: flex; align-items: start; justify-content: center">
  <img src='{{"/assets/img/deterministic-differentiable-pruning/deterministic-differentiable-pruning-table10.webp" | relative_url}}' width="49%">
  <div style="flex-grow: 0; width: 2%;"></div>
  <img src='{{"/assets/img/deterministic-differentiable-pruning/deterministic-differentiable-pruning-table11.webp" | relative_url}}' width="49%">
</div>