---
title: "[논문리뷰] SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm"
last_modified_at: 2026-09-13
categories:
  - 논문리뷰
tags:
  - NLP
  - ICML
excerpt: "SiameseNorm 논문 리뷰 (ICML 2026)"
use_math: true
classes: wide
---

> ICML 2026. [[Paper](https://arxiv.org/abs/2602.08064)] [[Github](https://github.com/Qwen-Applications/SiameseNorm)]  
> Tianyu Li, Dongchen Han, Zixuan Cao, Haofeng Huang, Mengyu Zhou, Ming Chen, Erchao Zhao, Xiaoxi Jiang, Guanjun Jiang, Gao Huang  
> Tsinghua University | Qwen Large Model Application Team, Alibaba  
> 8 Feb 2026  

<center><img src='{{"/assets/img/siamese-norm/siamese-norm-fig1.webp" | relative_url}}' width="90%"></center>

## Introduction
Pre-Norm은 최신 Transformer 모델을 매우 큰 규모로 scaling하는 데 놀라운 성공을 거두었지만, 정규화 패러다임 선택은 여전히 ​​해결되지 않은 문제이다. 특히 최근 연구에 따르면 Pre-Norm 모델에서 깊은 layer의 상당 부분을 pruning해도 성능 저하는 미미한 수준에 그치는 경우가 많다. 이는 Pre-Norm이 뛰어난 학습 안정성을 제공하지만, layer 활용도와 유효 깊이가 제한적일 수 있음을 시사한다.

반면 Post-Norm은 안정적으로 최적화될 경우 더 나은 최종 성능을 보이는 경우가 많다. 그러나 모델 크기가 커질수록 Post-Norm Transformer의 안정적이고 효율적인 학습을 보장하는 것은 점점 더 어려워진다. 또한, Post-Norm 기반 모델은 hyperparameter에 매우 민감하다. 따라서 표준 Pre-Norm 학습 방법을 Post-Norm 아키텍처에 직접 적용하면 종종 발산하거나 최적 성능을 달성하지 못한다.

자연스러운 전략 중 하나는 Pre-Norm과 Post-Norm을 결합하여 두 기법의 상호 보완적인 강점을 활용하는 것이다. 그러나 실제로는 이러한 하이브리드 설계가 특정 환경 외에서는 학습 안정성이 부족한 경우가 많으며, 이는 Post-Norm의 불안정성과 유사하다. 저자들은 이러한 현상이 두 패러다임 사이의 구조적 긴장 관계에 있다고 생각하였다. Pre-Norm은 신호 크기가 자연스럽게 증가할 수 있도록 identity 경로를 유지함으로써 대규모 모델을 안정화하는 반면, Post-Norm은 각 residual을 더한 후 신호를 조절하여 이러한 증가를 제한한다. 따라서 단일 스트림 아키텍처 내에서는 Pre-Norm과 Post-Norm이 원하는 속성을 조화시키기가 어렵다.

이러한 통찰력을 바탕으로, 본 논문에서는 Pre-Norm과 Post-Norm을 통합한 2개의 스트림으로 구성된 residual 아키텍처인 **SiameseNorm**을 제안하였다. SiameseNorm은 공유 연산 모듈을 갖는 두 개의 결합된 residual 스트림을 유지한다. 하나는 Pre-Norm의 identity-gradient 경로를 보존하는 비정규화 스트림이고, 다른 하나는 Post-Norm의 표현 역학을 유지하는 정규화 스트림이다. Identity-gradient 전파를 정규화된 표현 학습에서 분리하고, 각 연산 모듈 전에 두 스트림의 정규화된 표현을 더하여 두 스트림을 융합함으로써, SiameseNorm은 두 패러다임의 장점을 최소한의 오버헤드로 결합한다.

SiameseNorm의 가장 큰 장점은 기존 Transformer 구성과의 호환성이다. 모델 유형과 모달리티에 관계없이 SiameseNorm은 최적화의 robustness를 유지하면서 Pre-Norm baseline 대비 일관되게 성능을 향상시킨다.

## Theoretical Motivation
### 1. Pre-Norm
Pre-Norm 패러다임은 Layer Normalization (LN)을 residual 분기 내에만 배치한다.

$$
\begin{equation}
X_{i+1} = X_i + F_i (\textrm{LN}_i (X_i))
\end{equation}
$$

위 식에서 $$\theta_i$$에 대한 loss gradient는 다음과 같다.

$$
\begin{aligned}
\nabla_{\theta_i} \mathcal{L} &= \frac{\partial \mathcal{L}}{\partial X_N} \left( \prod_{j=N-1}^{i+1} \frac{\partial X_{j+1}}{\partial X_j} \right) \frac{\partial X_{i+1}}{\partial \theta_i} \\
&= \frac{\partial \mathcal{L}}{\partial X_N} \left[ \prod_{j=N-1}^{i+1} \left( \textbf{I} + \frac{\partial F_j (\textrm{LN}_j (X_j))}{\partial X_j} \right) \right] \frac{\partial X_{i+1}}{\partial \theta_i} \\
&= \frac{\partial \mathcal{L}}{\partial X_N} \left[ \prod_{j=N-1}^{i+1} \left( \textbf{I} + \textbf{J}_{F_j} \textbf{J}_{\textrm{LN}_j} \right) \right] \frac{\partial X_{i+1}}{\partial \theta_i}
\end{aligned}
$$

여기서 곱셈은 $N-1$번째 layer부터 $i+1$번째 layer까지의 Jacobian 행렬의 합성을 나타낸다. Skip connection에 의해 유도되는 identity 항 $\textbf{I}$는 명시적인 gradient 경로를 제공하며, 이는 Pre-Norm의 최적화 안정성에 중요한 이유이다. 그러나 magnitude가 무한히 증가할 수 있다. 이로 인해 더 깊은 layer에서 스케일 불일치가 발생한다. 각 block은 정규화된 입력을 받지만 점점 더 커지는 메인 경로를 업데이트해야 한다. 결과적으로 더 깊은 block의 상대적 기여도가 희석되어 Pre-Norm의 유효 깊이가 제한된다.

### 2. Post-Norm
Post-Norm 패러다임은 residual을 더한 후에 LN을 적용한다.

$$
\begin{equation}
X_{i+1} = \textrm{LN}_i (X_i + F_i (X_i))
\end{equation}
$$

PostNorm은 메인 경로에 정규화를 적용함으로써 hidden 표현의 스케일을 깊이에 따라 일정하게 유지하여 각 block이 PreNorm보다 더 강한 영향을 미칠 수 있도록 한다. 그러나 이러한 이점은 backpropagation 과정에서 각 layer의 gradient에 LN Jacobian을 곱해야 하므로 최적화 불안정성을 수반한다.

$$
\begin{aligned}
\nabla_{\theta_i} \mathcal{L} &= \frac{\partial \mathcal{L}}{\partial X_N} \left( \prod_{j=N-1}^{i+1} \frac{\partial X_{j+1}}{\partial X_j} \right) \frac{\partial X_{i+1}}{\partial \theta_i} \\
&= \frac{\partial \mathcal{L}}{\partial X_N} \left[ \prod_{j=N-1}^{i+1} \textbf{J}_{\textrm{LN}_j} (\textbf{I} + \textbf{J}_{F_j}) \right] \frac{\partial X_{i+1}}{\partial \theta_i}
\end{aligned}
$$

$$\textbf{I} + \textbf{J}_{F_j}$$가 well-conditioned matrix라도, $$\textbf{J}_\textrm{LN}$$을 반복적으로 적용하면 곱셈적 불안정성이 발생한다. 이러한 누적 효과로 인해 깊이 $N$이 증가함에 따라 gradient가 사라지거나 폭발적으로 증가할 수 있다. 이러한 메커니즘이 layer가 많은 Post-Norm Transformer에서 관찰되는 심각한 최적화 불안정성을 설명한다.

### 3. Structural Tension
기존의 하이브리드 방식은 Pre-Norm과 Post-Norm 동작을 서로 다른 layer 또는 submodule에 할당함으로써 이러한 긴장을 부분적으로 완화하였다. 그러나 모든 업데이트가 여전히 하나의 공유 메인 경로를 따라 누적되기 때문에 동일한 표현 방식이 두 가지 상충되는 역할을 동시에 수행해야 한다. 즉, 안정적인 최적화를 위해 정규화되지 않은 identity-gradient 경로를 유지하면서, residual의 magnitude를 제어하기 위해 반복적인 정규화를 적용해야 한다. 이 두 가지 요구 사항을 하나의 스트림 내에서 충족하기는 어렵다. 따라서 하나의 residual 경로 내에서 Pre-Norm과 Post-Norm 연산을 직접 혼합하는 대신, 이러한 역할을 구조적으로 분리하되 서로 연결된 스트림으로 구성하는 것이 본 논문의 목표이다.

## Method
SiameseNorm은 $X_i$와 $Y_i$로 표시되는 두 개의 연결된 residual 스트림을 유지한다. $X_i$ 스트림은 각 residual 업데이트 후 정규화되며, 이는 hidden-state magnitude를 제어하는 ​​Post-Norm 경로와 유사하다. $Y_i$ 스트림은 메인 경로에서 정규화 없이 residual 업데이트를 누적하며, 이는 안정적인 gradient 전파를 위한 identity 경로를 유지하는 Pre-Norm 경로와 유사하다. 각 layer에서 두 스트림은 공유 residual block $F_i$를 통해 상호작용하므로, 이 아키텍처는 무시할 수 있는 수준의 파라미터 오버헤드를 발생시킨다.

<center><img src='{{"/assets/img/siamese-norm/siamese-norm-algo1.webp" | relative_url}}' width="36%"></center>

##### 일반화 능력
SiameseNorm은 간단한 파라미터 구성을 통해 여러 정규화 패러다임을 연결한다. $$\textrm{LN}_i^X$$를 0으로 설정하면 Pre-Norm 토폴로지가 복원되고, $$\textrm{LN}_i^Y$$를 0으로 설정하면 Post-Norm 스타일의 아키텍처가 생성된다. 계층별 스트림 선택 기능은 초기 layer는 Post-Norm을, 나중 layer는 Pre-Norm을 사용하는 [Mix-LN](https://arxiv.org/abs/2412.13795)과 같은 하이브리드 스위칭 방식을 지원한다. 따라서 SiameseNorm은 Pre-Norm, Post-Norm, 하이브리드 방식의 동작을 포괄한다.

##### Gradient 분석
$$S_i = [X_i, Y_i]^\top$$를 두 스트림의 연결된 state라고 하자. Residual 변환 파라미터 $$\theta_i$$에 대한 loss gradient는 다음과 같다.

$$
\begin{aligned}
\nabla_{\theta_i} \mathcal{L} &= \frac{\partial \mathcal{L}}{\partial S_N} \left( \prod_{j=N-1}^{i+1} \frac{\partial S_{j+1}}{\partial S_j} \right) \frac{\partial S_{i+1}}{\partial O_i} \frac{\partial O_i}{\partial \theta_i} \\
&= \frac{\partial \mathcal{L}}{\partial S_N} \left( \prod_{j=N-1}^{i+1} \frac{\partial S_{j+1}}{\partial S_j} \right) \begin{bmatrix} \textbf{J}_{\textrm{LN}_i^X} \\ \textbf{I} \end{bmatrix} \frac{\partial O_i}{\partial \theta_i}
\end{aligned}
$$

Block Jacobian transition matrix는 다음과 같다.

$$
\begin{equation}
\frac{\partial S_{j+1}}{\partial S_j} = \begin{bmatrix} \textbf{J}_{\textrm{LN}_j^X} (\textbf{I} + \textbf{J}_{F_j}) & \textbf{J}_{\textrm{LN}_j^X} \textbf{J}_{F_j} \textbf{J}_{\textrm{LN}_j^Y} \\ \textbf{J}_{F_j} & \textbf{I} + \textbf{J}_{F_j} \textbf{J}_{\textrm{LN}_j^Y} \end{bmatrix}
\end{equation}
$$

이 transition matrix의 diagonal block들은 두 가지 정규화 패러다임 모두와의 구조적 연결을 보여준다. 오른쪽 아래 block은 Pre-Norm transition과 일치하여 $Y$ 스트림을 통해 직접적인 identity-gradient 경로를 제공한다. 왼쪽 위 block은 Post-Norm transition과 유사하며 $X$ 스트림을 통해 정규화된 residual 경로를 생성한다. 두 스트림 모두 동일한 residual 업데이트 $O_i$를 공유하므로 각 residual block은 무시할 수 있는 오버헤드로 두 경로 모두에서 최적화 신호를 받을 수 있다.

##### 보조 메커니즘
SiameseNorm을 기존 학습 레시피와 호환시키기 위해 두 가지 보조 메커니즘을 도입하였다.

1. **Normalized Input**: 공유 residual block 전에 집계된 표현에 추가적인 LN을 적용하여 표준 Transformer와 일관된 안정적인 입력 분포를 보장한다.
2. **Depth-wise Scaling**: PostNorm 스트림으로 전송되는 residual 업데이트를 $1/\sqrt{l+1}$로 scaling한다 ($l$은 layer 인덱스).

최적화 관점에서, 이 depth-wise scaling은 PostNorm 스타일 경로의 민감도를 줄이고 초기 업데이트 크기를 안정적인 PreNorm 학습 레시피와 더 잘 일치시켜 더 높은 learning rate를 사용할 수 있도록 한다. 또한, 깊이가 증가함에 따라 두 스트림 간에 자연스럽게 scale 불일치가 발생한다. PreNorm 스트림의 hidden-state norm은 증가하는 경향이 있는 반면, PostNorm 스트림은 정규화에 의해 제한된 state를 유지한다. 이는 깊은 layer에서 딜레마를 야기한다. 공유 residual 업데이트가 너무 작기 때문에 점점 증가하는 Pre-Norm 스트림에 의미 있는 영향을 미치지 못할 수 있지만, 제한된 스트림의 안정성을 유지하기에는 너무 클 수 있다. Depth-wise scaling은 제한된 스트림에 주입되는 업데이트를 감쇠시켜 이러한 불일치를 완화하고, 학습 안정성을 유지하면서 두 경로의 기여도를 재균형화한다.

이러한 실질적인 수정 사항들은 residual Jacobian $$\textbf{J}_{F_j}$$의 내부 정의와 scale에만 영향을 미치고, 두 흐름 전환 구조는 그대로 유지한다.

##### 계산 오버헤드
SiameseNorm은 보조적인 LN 연산만 도입하므로 Transformer에서 무시할 수 있는 수준의 오버헤드만 발생한다. LN은 파라미터와 계산량 측면에서 주요 attention 및 MLP block에 비해 가볍기 때문에 추가적인 정규화 연산은 전체 복잡성에 미미한 영향만 미친다. 이론적으로 파라미터 개수와 FLOPs는 0.1% 미만으로 증가한다. 실제적으로 15B MoE 모델로 scaling했을 때 학습 속도는 0.5%만 감소하고 활성화 메모리는 2%만 증가한다.

## Experiments
### 1. Main Results
다음은 1.3B 모델과 15A2B MoE 모델에 대한 평가 결과이다.

<center><img src='{{"/assets/img/siamese-norm/siamese-norm-table1.webp" | relative_url}}' width="90%"></center>

### 2. Generality Across Depths and Modalities
다음은 다양한 모델 깊이와 모달리티에 대한 비교 결과이다.

<center><img src='{{"/assets/img/siamese-norm/siamese-norm-table2.webp" | relative_url}}' width="45%"></center>

### 3. Ablation
다음은 [HybridNorm](https://arxiv.org/abs/2503.04598), [ResiDual](https://arxiv.org/abs/2304.14802)과 학습 loss 그래프를 비교한 결과이다.

<center><img src='{{"/assets/img/siamese-norm/siamese-norm-fig4.webp" | relative_url}}' width="75%"></center>
<br>
다음은 주요 구성 요소에 대한 ablation study 결과이다.

<center><img src='{{"/assets/img/siamese-norm/siamese-norm-table3.webp" | relative_url}}' width="46%"></center>

### 4. Analysis
다음은 gradient norm을 비교한 결과이다.

<center><img src='{{"/assets/img/siamese-norm/siamese-norm-fig5.webp" | relative_url}}' width="52%"></center>
<br>
다음은 두 스트림의 스케일 비율을 비교한 결과이다.

<center><img src='{{"/assets/img/siamese-norm/siamese-norm-fig6.webp" | relative_url}}' width="82%"></center>