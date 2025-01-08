---
title: "[논문리뷰] Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model"
last_modified_at: 2025-01-09
categories:
  - 논문리뷰
tags:
  - Diffusion
  - Transformer
  - Computer Vision
  - NLP
  - AI
  - Meta
excerpt: "Transfusion 논문 리뷰"
use_math: true
classes: wide
---

> arXiv 2024. [[Paper](https://arxiv.org/abs/2408.11039)]  
> Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, Omer Levy  
> Meta | Waymo | University of Southern California  
> 20 Aug 2024  

<center><img src='{{"/assets/img/transfusion/transfusion-fig1.webp" | relative_url}}' width="90%"></center>

## Introduction
본 논문에서는 하나의 모델을 학습시켜 discrete한 텍스트 토큰을 예측하고 continuous한 이미지를 diffuse함으로써 정보 손실 없이 두 모달리티를 완전히 통합할 수 있음을 보여주며, discrete하고 continuous한 모달리티를 원활하게 생성할 수 있는 모델을 학습하기 위한 레시피인 **Transfusion**을 소개한다. 

각 모달리티에 대해 다른 목적, 즉 텍스트의 next token prediction과 이미지의 diffusion을 사용하여 50% 텍스트 데이터와 50% 이미지 데이터에 대한 Transformer 모델을 사전 학습시킨다. 모델은 학습 시 두 모달리티와 loss function에 모두 노출된다. 텍스트 토큰은 임베딩 행렬을 통해 벡터로 변환되며, 각 이미지는 패치 벡터들의 시퀀스로 나타낸다. 저자들은 텍스트 토큰에 casual attention을 적용하고 이미지 패치에 bidirectional attention을 적용하였다. Inference의 경우, 언어 모델에서 텍스트를 생성하고 diffusion model에서 이미지를 생성하는 표준 관행을 결합하는 디코딩 알고리즘을 도입하였다.

[Chameleon](https://arxiv.org/abs/2405.09818)과 비교했을 떄, Transfusion은 모든 모달리티 조합에서 더 나은 확장성을 보였으며, text-to-image 생성에서는 1/3 미만의 계산량으로 FID와 CLIP score를 능가하였다. Transfusion은 동일한 FLOP에서 Chameleon 모델보다 약 2배 낮은 FID를 달성하였다. Image-to-text 생성의 경우, Transfusion은 Chameleon의 21.8%에 해당하는 FLOP으로 동일한 성능을 보였다. Text-to-text 예측의 경우, Chameleon의 약 50%~60%에 해당하는 FLOP으로 동일한 perplexity parity를 달성하였다.

Transfusion에서 이미지 내 bidirectional attention을 casual attention으로 대체하면 text-to-image 생성에 해가 된다. 또한 U-Net up/down block을 추가하여 이미지를 디코딩/인코딩하면 성능에 대한 적은 손실로 더 큰 이미지 패치를 압축하여 비용을 최대 64배까지 줄일 수 있다.

Transfusion 7B 모델은 DALL-E 2나 SDXL과 같은 다른 인기 모델보다 성능이 뛰어나며, 텍스트 벤치마크에서 Llama 1과 동일한 수준의 텍스트 벤치마크 성능을 달성하였다. 

## Method
Transfusion은 discrete하고 continuous한 모달리티를 모두 이해하고 생성하기 위해 하나의 통합 모델을 학습시키는 방법이다. 

### 1. Data Representation
저자들은 두 가지 모달리티, 즉 discrete한 텍스트와 continuous한 이미지에 걸친 데이터로 실험하였다. 각 텍스트 문자열은 고정된 vocabulary에서 discrete한 토큰 시퀀스로 tokenize되며, 각 토큰은 정수로 표현된다. 각 이미지는 VAE를 사용하여 latent 패치들로 인코딩되며, 각 패치는 continuous한 벡터로 표현된다. 패치는 왼쪽에서 오른쪽으로, 위에서 아래로 시퀀싱되어 각 이미지에서 패치 벡터 시퀀스를 만든다. 

두 모달리티를 함께 사용하는 경우, 텍스트 시퀀스에 삽입하기 전에 각 이미지 시퀀스를 특수한 BOI 토큰과 EOI 토큰으로 둘러싸며, 이를 통해 텍스트 토큰을 나타내는 정수와 이미지 패치를 나타내는 벡터를 모두 포함할 수 있는 하나의 시퀀스가 된다.

### 2. Model Architecture
<center><img src='{{"/assets/img/transfusion/transfusion-fig3.webp" | relative_url}}' width="35%"></center>
<br>
모델 파라미터의 대부분은 모달리티에 관계없이 모든 시퀀스를 처리하는 하나의 transformer에 속한다. Transformer는 $\mathbb{R}^d$의 고차원 벡터 시퀀스를 입력으로 받고 유사한 벡터를 출력으로 생성한다. 

데이터를 이 공간으로 변환하기 위해, 가벼운 모달리티별 구성 요소를 사용한다. 텍스트의 경우, 임베딩 행렬을 통해 각 입력 정수를 벡터 공간으로 변환하고 각 출력 벡터를 vocabulary에 대한 discrete한 분포로 변환한다. 이미지의 경우, $k \times k$ 패치 벡터들의 local window를 하나의 transformer 벡터로 압축하기 위한 두 가지 옵션을 실험하였다. 

1. 단순한 linear layer (timestep의 임베딩을 모든 패치 벡터에 더한 후 입력)
2. U-Net의 up/down block (AdaLayerNorm를 일반적인 layer norm으로 교체)

### 3. Transfusion Attention
언어 모델은 일반적으로 casual attention을 사용하여 미래 토큰의 정보를 누출하지 않고 전체 시퀀스에 대한 loss와 gradient를 효율적으로 계산한다. 반면, 이미지는 일반적으로 제한 없는 bidirectional attention으로 모델링된다. 

<center><img src='{{"/assets/img/transfusion/transfusion-fig4.webp" | relative_url}}' width="40%"></center>
<br>
Transfusion은 두 attention 패턴을 결합하기 위해 시퀀스의 모든 요소에 casual attention을 적용하고 각 개별 이미지의 요소 내에서 bidirectional attention을 적용한다. 이를 통해 모든 이미지 패치는 동일한 이미지 내의 다른 모든 패치에 attention할 수 있지만 시퀀스에 이전에 나타난 다른 이미지의 텍스트나 패치에만 attention할 수 있다. 이미지 내 attention을 사용하면 모델 성능이 크게 향상된다. 

### 4. Training Objective
모델을 학습시키기 위해, 언어 모델링 loss $$\mathcal{L}_\textrm{LM}$$을 텍스트 토큰의 예측에 적용하고 diffusion loss $$\mathcal{L}_\textrm{DDPM}$$을 이미지 패치의 예측에 적용한다. LM loss는 토큰별로 계산되고, diffusion loss는 이미지별로 계산된다. 구체적으로, diffusion process에 따라 각 입력 latent 이미지 $x_0$에 noise $\epsilon$을 추가하여 patchification 전에 $x_t$를 생성한 다음, 이미지 레벨의 diffusion loss를 계산한다. 전체 loss는 다음과 같다. 

$$
\begin{equation}
\mathcal{L}_\textrm{Transfusion} = \mathcal{L}_\textrm{LM} + \lambda \cdot \mathcal{L}_\textrm{DDPM}
\end{equation}
$$

즉, discrete한 분포의 loss와 continuous한 분포의 loss를 결합하여 동일한 모델을 최적화한다. 

### 5. Inference
디코딩 알고리즘도 LM과 diffusion 두 가지 모드 사이를 전환한다. LM 모드에서는 예측된 분포에서 토큰별로 샘플링하는 표준 관행을 따른다. BOI 토큰을 샘플링하면 디코딩 알고리즘이 diffusion 모드로 전환되고, 여기서는 diffusion model에서 디코딩하는 표준 절차를 따른다. 

구체적으로, $n$개의 이미지 패치 형태로 순수 noise $x_T$를 입력 시퀀스에 추가하고 $T$ step에 걸쳐 noise를 제거한다. 각 timestep $t$에서 noise 예측을 사용하여 $x_{t−1}$을 생성한 다음 시퀀스에서 $x_t$를 덮어쓴다. 즉, 모델은 항상 noise가 적용된 이미지의 마지막 timestep을 조건으로 하며 이전 timestep에는 attention할 수 없다. 

Diffusion process가 끝나면 예측된 이미지에 EOI 토큰을 추가하고 LM 모드로 다시 전환한다. 이 알고리즘을 사용하면 텍스트와 이미지 모달리티를 혼합하여 생성할 수 있다.

## Experiments
- 데이터
  - 두 모달리티에서 1:1 비율로 총 5천억 개의 토큰을 샘플링
  - 텍스트: Llama 2 corpus (토큰 2조 개)
  - 이미지: 캡션이 있는 Shutterstock 이미지 3.8억 개 (256$\times$256)
- VAE
  - [VQ-GAN](https://arxiv.org/abs/2012.09841)을 따라 86M 모델을 100만 step동안 학습
  - CNN 인코더/디코더로 구성
  - 256$\times$256$\times$3 $\rightarrow$ 32$\times$32$\times$8
  - codebook 토큰 수: 16,384
- 학습 디테일
  - optimizer: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$)
  - learning rate: $3 \times 10^{-4}$, 4천 step warm-up, cosine scheduler를 따라 $1.5 \times 10^{-5}$로 감소
  - batch size: 200만 토큰 (한 시퀀스에 4096 토큰)
  - iteration: 25만 step
  - weight decay = 0.1, clip gradients = 1.0

(왼쪽) 모델 평가 방법과 (오른쪽) 모델의 구성은 아래 표와 같다. 

<div style="display: flex; align-items: end; justify-content: center">
  <img src='{{"/assets/img/transfusion/transfusion-table1.webp" | relative_url}}' width="43%">
  <div style="flex-grow: 0; width: 3%;"></div>
  <img src='{{"/assets/img/transfusion/transfusion-table2.webp" | relative_url}}' width="32%">
</div>

### 1. Controlled Comparison with Chameleon
다음은 다양한 크기의 Transfusion과 [Chameleon](https://arxiv.org/abs/2405.09818)을 다양한 벤치마크에서 비교한 그래프이다. 

<center><img src='{{"/assets/img/transfusion/transfusion-fig5.webp" | relative_url}}' width="85%"></center>
<br>
다음은 5천억 개의 토큰으로 학습된 7B Transfusion과 Chameleon을 비교한 표이다. Parity FLOP Ratio는 Chameleon 7B와 동일한 성능을 내기 위한 Transfusion FLOP의 비율이다. 

<center><img src='{{"/assets/img/transfusion/transfusion-table3.webp" | relative_url}}' width="64%"></center>
<br>
다음은 텍스트 벤치마크에서 0.76B Transfusion과 Chameleon을 비교한 표이다. 

<center><img src='{{"/assets/img/transfusion/transfusion-table4.webp" | relative_url}}' width="79%"></center>

### 2. Architecture Ablations
다음은 이미지 내의 bidirectional attention 유무에 대한 ablation 결과이다. (0.76B Transfusion, 2$\times$2 패치)

<center><img src='{{"/assets/img/transfusion/transfusion-table5.webp" | relative_url}}' width="67%"></center>
<br>
다음은 패치 크기에 대한 ablation 결과이다. (0.76B Transfusion)

<center><img src='{{"/assets/img/transfusion/transfusion-table6.webp" | relative_url}}' width="80%"></center>
<br>
다음은 인코더/디코더에 대한 ablation 결과이다. (2$\times$2 패치)

<center><img src='{{"/assets/img/transfusion/transfusion-table7.webp" | relative_url}}' width="75%"></center>
<br>
다음은 캡션 앞에 이미지가 나타날 때 샘플링된 diffusion noise의 양을 최대 $t = 500$으로 제한한 경우와 제한하지 않은 경우의 Transfusion 성능을 비교한 표이다. (U-Net 인코더/디코더, 2$\times$2 패치)

<center><img src='{{"/assets/img/transfusion/transfusion-table8.webp" | relative_url}}' width="62%"></center>

### 3. Comparison with Image Generation Literature
다음은 2조 개의 토큰으로 학습된 7B Transfusion으로 생성한 이미지들이다. (U-Net 인코더/디코더, 2$\times$2 패치)

<center><img src='{{"/assets/img/transfusion/transfusion-fig2.webp" | relative_url}}' width="85%"></center>
<br>
다음은 2조 개의 토큰으로 학습된 7B Transfusion을 비슷한 크기의 모델들과 비교한 표이다. ($\vphantom{1}^\ast$는 텍스트 인코더의 파라미터를 고정한 경우)

<center><img src='{{"/assets/img/transfusion/transfusion-table9.webp" | relative_url}}' width="78%"></center>

### 4. Image Editing
다음은 fine-tuning된 7B Transfusion으로 편집한 이미지들이다. 

<center><img src='{{"/assets/img/transfusion/transfusion-fig6.webp" | relative_url}}' width="85%"></center>