---
title: "[논문리뷰] GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation"
last_modified_at: 2026-08-10
categories:
  - 논문리뷰
tags:
  - Gaussian Splatting
  - 3D Reconstruction
  - 3D Vision
  - ECCV
excerpt: "GaussianGPT 논문 리뷰 (ECCV 2026)"
use_math: true
classes: wide
---

> ECCV 2026. [[Paper](https://arxiv.org/abs/2603.26661)] [[Page](https://nicolasvonluetzow.github.io/GaussianGPT/)] [[Github](https://github.com/nicolasvonluetzow/gaussiangpt)]  
> Nicolas von Lützow, Barbara Rössle, Katharina Schmid, Matthias Nießner  
> Technical University of Munich  
> 27 Mar 2026  

<center><img src='{{"/assets/img/gaussian-gpt/gaussian-gpt-fig1.webp" | relative_url}}' width="80%"></center>

## Introduction
본 논문에서는 구조화된 장면 primitive에 대한 순차적 예측으로 3D 장면을 합성하는 autoregressive한 장면 생성 및 완성 방법인 **GaussianGPT**를 소개한다. 장면을 vector-quantize된 3D Gaussian에서 파생된 discrete한 토큰 시퀀스로 표현함으로써, transformer 기반 모델이 다음 토큰 예측을 통해 장면을 점진적으로 생성, 확장, 편집할 수 있도록 한다. 이는 명시적인 3D 표현과 autoregressive transformer의 inductive bias를 결합하여 diffusion 기반 패러다임에 대한 보완적인 대안을 제시한다.

## Method
본 논문의 목표는 GPT 스타일의 transformer를 사용한 autoregressive한 생성을 통해 3D Gaussian 장면을 합성하는 것이다. GPT 모델은 discrete한 토큰 시퀀스에서 작동하고 컨텍스트 범위가 제한적이기 때문에, 먼저 3D 장면을 간결하고 discrete한 형태로 표현해야 한다.

<center><img src='{{"/assets/img/gaussian-gpt/gaussian-gpt-fig2.webp" | relative_url}}' width="100%"></center>
<br>
이를 위해, Gaussian 장면을 discrete latent grid 표현으로 매핑하고 이를 충실하게 재구성하는 sparse 3D convolution 오토인코더를 활용한다. 이 표현을 기반으로 latent grid를 토큰 시퀀스로 직렬화(serialize)하고, 그리드 occupancy와 feature의 결합 분포를 모델링하는 causal transformer를 학습시킨다.

### 1. Scene Compression via Sparse 3D Latent Encoding
토큰 기반 autoregressive 모델링을 구현하기 위해, 먼저 continuous한 3D Gaussian 장면을 압축된 discrete latent grid 표현으로 변환한다. 본 논문의 압축 단계는 다음과 같이 세 단계로 구성된다.

1. Gaussian을 sparse 3D feature grid로 projection
2. Sparse 3D convolution 오토인코더를 사용하여 이 그리드를 인코딩
3. Latent 표현을 vector quantization

##### Sparse 3D Feature Grid
3DGS 알고리즘에 따라, 입력은 Gaussian 집합으로 표현되는 장면이다. 각 Gaussian은 위치, 불투명도, 크기, 회전, 색상과 같은 continuous한 속성 집합으로 정의된다. 월드 좌표계로 그리드를 정의하고, 각 위치에 따라 해당 voxel에 Gaussian을 할당한다. Gaussian 위치를 voxel 중심로부터의 상대적 offset으로 대체하고, 하나의 voxel에 여러 개의 Gaussian이 존재하는 경우 무작위로 서브샘플링하여 입력 3D Gaussian 장면을 얻는다.

각 voxel에 대해, 가벼운 인코딩 head를 사용하여 각 Gaussian feature를 인코딩한 다음, 결과를 통합 벡터로 concat하여 sparse한 입력 feature grid를 생성한다. 대칭적으로, 3D CNN을 통과한 후, feature별 디코딩 head를 사용하여 예측된 voxel별 feature 벡터를 개별 Gaussian 속성으로 다시 변환한다.

##### Sparse 3D CNN Encoder-Decoder
입력 feature grid는 [L3DG](https://arxiv.org/abs/2410.13530)를 따르는 sparse 3D convolution 인코더 $\mathcal{E}$와 디코더 $\mathcal{D}$에 의해 처리된다. 인코더는 그리드를 점진적으로 다운샘플링하여 압축된 latent 표현을 생성하고, 디코더는 voxel 수준의 feature를 재구성한다. Convolution 디자인은 공간적 locality와 변환 등변성을 유지하여 후속 생성 모델링에 적합한 구조화된 latent feature를 생성한다.

##### Vector Quantization
L3DG와 달리, 본 논문에서는 codebook 활용도와 품질을 향상시키는 것으로 입증된 [lookupfree quantization (LFQ)](https://kimjy99.github.io/논문리뷰/magvit-v2/)를 채택하였다. LFQ 개념에 따라, 본 논문에서 제시하는 인코더의 출력 $\textbf{z}$는 부호에 따라 0 또는 1로 discretize되어 codebook 인덱스에 직접적으로 대응한다.

##### 학습
네트워크는 re-rendering loss, occupancy loss, codebook loss를 조합하여 무작위로 선택된 이미지로 학습된다. Re-rendering loss로는 L1 RGB loss $$\mathcal{L}_\textrm{RGB}$$와 VGG19 기반 perceptual loss $$\mathcal{L}_\textrm{perc}$$를 사용하며, 두 loss 모두 샘플링된 이미지와 포즈 세트에 걸쳐 집계된다. 디코더 upsampling layer의 occupancy 예측은 L3DG를 따라 binary cross-entropy loss $$\mathcal{L}_\textrm{occ}$$를 기반으로 한다. Codebook loss $$\mathcal{L}_\textrm{LFQ}$$는 codebook 엔트로피를 증가시켜 codebook 사용률을 높이는 것을 목표로 한다. 

$$
\begin{equation}
\mathcal{L} = \lambda_\textrm{RGB} \mathcal{L}_\textrm{RGB} + \lambda_\textrm{perc} \mathcal{L}_\textrm{perc} + \lambda_\textrm{occ} \mathcal{L}_\textrm{occ} + \lambda_\textrm{LFQ} \textrm{softplus}(\mathcal{L}_\textrm{LFQ} + 5)
\end{equation}
$$

### 2. Autoregressive Modeling of Latent 3D Grids
Quantization된 latent grid가 주어졌을 때, 본 논문의 목표는 시퀀스 기반 autoregressive 모델을 사용하여 그리드의 occupancy 및 feature의 공동 분포를 모델링하는 것이다. 따라서 첫 번째 단계는 3D 그리드를 1D 토큰 시퀀스로 직렬화(serialize)하는 것이며, 그 후 causal transformer를 학습시켜 패턴을 모델링할 수 있다. 또한, 위치와 feature의 vocabulary를 분리하고 3D 위치 임베딩을 주입하여 모델에 prior를 제공한다.

##### 3D Grid Serialization
간단하고 고정된 $xyz$ 순회 순서를 사용하여 3D 구조를 1D 시퀀스로 선형화한다. $z$축이 최하위 차원이므로, 모든 $(x, y)$ 위치에서 장면 높이만큼의 열을 순회한 후 다음 위치로 이동한다. 이 패턴은 1D 시퀀스에서 3D locality를 유지하지는 않지만, 간단하고 해석하기 쉬운 순서라는 장점이 있다.

Voxel의 순서와 상대적 위치 인덱스를 이용하여 위치 토큰과 feature 토큰을 인터리빙하여 시퀀스를 구성한다. 가능한 시퀀스의 수는 장면 크기에 따라 세제곱으로 증가하므로, 전체 장면이 아닌 청크 단위로 GPT 모델을 작동시켜 필요한 컨텍스트 크기를 제한한다. 각 voxel에는 절대적인 위치가 아닌 현재 그리드 청크에 대한 상대적 위치 인덱스가 할당되므로, 모델은 로컬하게 작동하고 위치, 장면, 레이아웃 전반에 걸쳐 일반화할 수 있다.

##### Vocabulary 디자인
본 연구에서는 시퀀스를 공통 transformer backbone을 통해 처리하지만, 위치 토큰과 feature 토큰에 대해 별도의 vocabulary를 사용한다. 위치 토큰과 feature 토큰이 규칙적으로 교대로 나타나는 패턴에 따라, 각각 다른 위치 head와 feature head를 번갈아 사용한다. 위치 head는 다음에 점유될 voxel 인덱스를 예측하고, feature 헤드는 바로 앞 위치의 feature를 예측한다.

이러한 vocabulary와 예측의 명시적인 분리는 geometry와 외형 모델링을 분리하고, 공유 인덱스를 둘러싼 공간적 feature와 semantic feature 간의 경쟁을 방지한다. 또한, 이러한 디자인 덕분에 feature codebook의 크기와 관계없이 청크 크기와 그에 상응하는 위치 인덱스의 개수를 자유롭게 제어할 수 있다.

##### 3D Rotary Positional Encoding
GPT 스타일 모델에서 사용되는 transformer는 입력을 1차원 토큰 시퀀스로 처리하며, 위치에 대한 명시적인 개념을 필요로 한다. 일반적인 경우와 달리, 본 논문에서는 토큰 시퀀스가 ​​본질적으로 1차원 신호가 아니며, sparse 3D latent grid를 직렬화하여 얻는다. 직렬화된 순서에 표준 1D 위치 인코딩을 직접 적용하면 모델은 주로 시퀀스 근접성 개념을 학습하게 되는데, 이는 공간적 근접성으로 제대로 변환되지 않을 수 있다. 특히 $(x, y, z)$ 좌표가 가까운 voxel이 직렬화 과정에서는 멀리 떨어져 있을 수 있고, 그 반대의 경우도 마찬가지이다.

명시적인 공간적 inductive bias를 주입하기 위해, 3D RoPE를 사용하여 attention 메커니즘 내부에 실제 voxel 좌표를 인코딩한다. 이렇게 하면 attention score는 시퀀스 오프셋이 아닌 토큰 간의 상대적인 공간 오프셋의 함수가 된다. 이 접근 방식은 [TRELLIS.2](https://arxiv.org/abs/2512.14692)를 따르며, 이를 통해 모델은 직렬화 순서와 관계없이 공간적 locality를 추론할 수 있다.

또한, 시퀀스는 위치 토큰과 feature 토큰이 번갈아 나타나는데, 이 두 토큰은 모두 동일한 3D 위치에 대응한다. 따라서, 토큰 유형을 나타내는 네 번째 차원을 임베딩에 추가한다. 이 요소는 transformer가 혼합된 토큰 스트림에 대해 하나의 통합 attention 공식을 유지하면서 geometry와 외형을 더욱 명확하게 구분할 수 있도록 도와준다.

##### Transformer 아키텍처
본 논문의 decoder-only causal transformer는 GPT-2를 기반으로 하며, [nanochat](https://github.com/karpathy/nanochat)을 backbone으로 사용한다. 일반적인 GPT-2 파이프라인과 비교하여, 위치 임베딩 (3D RoPE), query-key normalization, per-layer residual scaling, Muon optimizer를 추가적으로 적용했다. nanochat backbone의 기본 구성에 포함되는 value 임베딩이나 sliding-window attention은 사용하지 않았다.

##### 학습
이 transformer는 표준 autoregressive objective를 사용하여 직렬화된 장면 토큰의 분포를 모델링하도록 학습된다. Tokenization된 장면 시퀀스 $$\textbf{t} = (t_1, \ldots, t_T)$$가 주어졌을 때, 모델은 이전의 모든 토큰을 조건으로 각 토큰을 예측한다.

$$
\begin{equation}
\mathcal{L}_\textrm{CE} = -\sum_{i=1}^T \log p_\theta (t_i \mid t_{< i})
\end{equation}
$$

학습은 teacher forcing 방식으로 진행된다. 표현 방식이 위치 토큰과 feature 토큰을 번갈아 사용하기 때문에 각 step에서 해당 vocabulary에 대한 cross-entropy loss를 계산한다. 유효하지 않은 vocabulary entry는 마스킹 처리하여 모델이 위치 단계에서는 위치 토큰만, feature 단계에서는 feature 토큰만 예측하도록 한다.

##### 장면 생성 및 완성
Inference 시, 장면은 이전에 생성된 컨텍스트를 기반으로 토큰을 샘플링하여 autoregressive 방식으로 생성된다. Transformer는 BOS 토큰에서 시작하여 EOS 토큰이 생성될 때까지 위치 토큰과 feature 토큰을 번갈아 예측한다.

Autoregressive 방식의 핵심 장점은 장면 완성과 생성이 동일한 메커니즘으로 처리된다는 점이다. 부분적인 장면이 주어지면, 관찰된 토큰들을 직렬화하여 모델이 생성을 계속할 수 있는 prefix 프롬프트로 사용한다. 이를 통해 transformer는 기존 장면 컨텍스트와 완벽하게 일관성을 유지하면서 누락된 geometry와 외형을 자연스럽게 추론할 수 있다. 동일한 원리를 통해 고정된 학습 청크 크기를 넘어 대규모 장면 합성도 가능하다. 이전에 생성된 토큰들을 컨텍스트로 사용하는 sliding window에 장면 완성을 반복적으로 적용함으로써 장면을 지속적으로 확장할 수 있다.

Autoregressive 생성의 구성적 특성 덕분에 현재 시퀀스를 기반으로 예측 범위를 더욱 제한할 수도 있다. 구체적으로, 각 위치 토큰은 현재 청크 내의 voxel 인덱스에 직접적으로 대응하므로, 이미 생성된 위치를 마스킹하여 샘플링된 시퀀스가 ​​항상 순서 제약 조건을 준수하도록 할 수 있다. 또한, 순차적 특성 덕분에 tree search가 가능하다. 이를 활용하여 점유되지 않은 열을 다시 샘플링함으로써 더 많은 점유가 이루어지고 연결된 장면을 생성하는 더 큰 장면을 만들 수 있다.

## Experiments
- 데이터셋: PhotoShape, 3D-FRONT, Aria Synthetic Environments
- 오토인코더
  - voxel 크기: base 0.025m $\rightarrow$ latent 0.2m
  - codebook 크기: 4,096
  - loss 가중치
    - object: 이미지 4장, $$\lambda_\textrm{RGB} = 7.5$$, $$\lambda_\textrm{perc} = 0.3$$, $$\lambda_\textrm{occ} = 1.0$$, $$\lambda_\textrm{LFQ} = 0.1$$
    - 장면: 이미지 12장, $$\lambda_\textrm{RGB} = 12.5$$, $$\lambda_\textrm{perc} = 0.1$$, $$\lambda_\textrm{occ} = 1.0$$, $$\lambda_\textrm{LFQ} = 0.1$$
  - GPU: RTX A6000 4개
  - effective batch size: 장면 8, object 24
  - optimizer: Adam
- Transformer
  - base model: 장면은 GPT-2 medium, object는 GPT-2 small
  - 샘플링: temperature 0.9, Nucleus Sampling ($p = 0.9$)
  - GPU: GH200 4개
  - effective batch size: 64
  - optimizer: AdamW + Muon

### 1. Shape Synthesis
다음은 shape 합성에 대한 비교 결과이다.

<center><img src='{{"/assets/img/gaussian-gpt/gaussian-gpt-fig3.webp" | relative_url}}' width="90%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/gaussian-gpt/gaussian-gpt-table1.webp" | relative_url}}' width="41%"></center>

### 2. Scene Synthesis
다음은 장면 합성에 대한 비교 결과이다.

<center><img src='{{"/assets/img/gaussian-gpt/gaussian-gpt-fig4.webp" | relative_url}}' width="95%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/gaussian-gpt/gaussian-gpt-table2.webp" | relative_url}}' width="52%"></center>
<br>
다음은 장면 청크 완성 결과들이다.

<center><img src='{{"/assets/img/gaussian-gpt/gaussian-gpt-fig6.webp" | relative_url}}' width="85%"></center>
<br>
다음은 autoregressive outpainting을 통해 12m×12m 장면을 합성한 결과들이다.

<center><img src='{{"/assets/img/gaussian-gpt/gaussian-gpt-fig5.webp" | relative_url}}' width="100%"></center>

### 3. Autoregressive Modeling Ablations
다음은 (왼쪽) 직렬화 전략과 (오른쪽) 디자인 선택에 대한 ablation 결과이다.

<div style="display: flex; align-items: start; justify-content: center">
  <img src='{{"/assets/img/gaussian-gpt/gaussian-gpt-table3.webp" | relative_url}}' width="29%">
  <div style="flex-grow: 0; width: 5%;"></div>
  <img src='{{"/assets/img/gaussian-gpt/gaussian-gpt-table4.webp" | relative_url}}' width="31%">
</div>