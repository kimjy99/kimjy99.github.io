---
title: "[논문리뷰] TransPixeler: Advancing Text-to-Video Generation with Transparency"
last_modified_at: 2025-05-01
categories:
  - 논문리뷰
tags:
  - Diffusion
  - Text-to-Video
  - Image-to-Video
  - Computer Vision
  - CVPR
excerpt: "TransPixeler 논문 리뷰 (CVPR 2025)"
use_math: true
classes: wide
---

> CVPR 2025. [[Paper](https://arxiv.org/abs/2501.03006)] [[Page](https://wileewang.github.io/TransPixar/)] [[Github](https://github.com/wileewang/TransPixar)] [[Demo](https://huggingface.co/spaces/wileewang/TransPixar)]  
> Luozhou Wang, Yijun Li, Zhifei Chen, Jui-Hsien Wang, Zhifei Zhang, He Zhang, Zhe Lin, Yingcong Chen  
> HKUST(GZ) | HKUST | Adobe Research  
> 6 Jan 2025  

<center><img src='{{"/assets/img/transpixeler/transpixeler-fig1.webp" | relative_url}}' width="100%"></center>

## Introduction
본 논문에서는 사전 학습된 동영상 모델을 확장하여 사전 학습된 모델의 원래 능력을 유지하면서 해당 알파 채널을 생성하는 방법을 살펴보았다. 저자들의 목표는 현재 RGBA 학습 데이터셋의 한계를 넘어 콘텐츠를 생성하는 것이다. 사전 학습된 생성 모델 가중치를 활용하면 dense한 예측에서 out-of-distribution이 크게 향상된다. 그러나 RGBA 동영상 생성의 맥락에서 이러한 접근 방식은 일반적으로 먼저 RGB 채널을 생성한 다음 별도의 알파 채널 예측이 필요하다. 결과적으로 정보는 RGB에서 알파로 단방향으로 흐르므로 두 프로세스가 거의 분리되어 있다. RGBA 동영상 데이터의 가용성이 제한되어 있기 때문에 이러한 불균형은 어려운 물체를 생성할 때 알파 예측이 충분하지 않다.

본 논문에서는 사전 학습된 RGB 동영상 모델을 효과적으로 조정하여 RGB 채널과 알파 채널을 동시에 생성하는 **TransPixeler**를 제안하였다. 저자들은 SOTA [DiT](https://kimjy99.github.io/논문리뷰/dit) 동영상 생성 모델을 활용하고, 알파 채널을 생성하기 위해 텍스트와 RGB 토큰 뒤에 새로운 토큰을 추가로 도입하였다. 수렴을 용이하게 하기 위해 알파 토큰의 위치 임베딩을 다시 초기화하고, 알파 토큰과 RGB 토큰을 구별하기 위해 0으로 초기화된 학습 가능한 도메인 임베딩을 도입하였다. 또한, 알파 토큰을 qkv 공간으로 projection하는 데만 적용되는 [LoRA](https://kimjy99.github.io/논문리뷰/lora) 기반 fine-tuning 체계를 채택하여 RGB 생성 품질을 유지하였다. 원래의 입출력 구조를 보존하고 LoRA를 통해 기존의 attention 메커니즘에 의존하면서 모달리티를 확장하였다.

확장된 시퀀스에는 텍스트, RGB, 알파 토큰이 포함되기 때문에 self-attention은 3$\times$3 grouped attention 행렬로 나뉜다. 저자들은 RGBA 생성을 위한 attention 메커니즘을 체계적으로 분석하였다. (A-attend-to-B는 A가 query, B가 key)

1. **Text-attend-to-RGB**, **RGB-attend-to-Text**: 텍스트와 RGB 토큰 간의 상호 작용은 원래 모델의 생성 능력을 나타낸다. 이러한 attention 계산 프로세스 중에 텍스트와 RGB 토큰에 미치는 영향을 최소화하면 원래 모델의 성능을 더 잘 유지할 수 있다. 
2. **RGB-attend-to-Alpha**: 기존 방법의 근본적인 한계는 이 attention 메커니즘이 부족하다는 것이다. 이 attention은 알파 정보를 기반으로 RGB 토큰을 정제하여 RGB-알파 정렬을 개선하는 데 필요하다. 
3. **Text-attend-to-Alpha**: 제한된 학습 데이터로 인해 발생하는 위험을 줄이기 위해 이 attention 메커니즘을 제거하여 모델의 성능을 저하시킬 수 있다. 이 제거는 또한 모델의 원래 능력을 보존하는 데 도움이 된다.

이러한 테크닉을 통합함으로써, TransPixeler은 강력한 RGB-알파 정렬을 유지하는 동시에 제한된 학습 데이터로 다양한 RGBA 생성을 달성하였다.

## Method
<center><img src='{{"/assets/img/transpixeler/transpixeler-fig3.webp" | relative_url}}' width="100%"></center>
<br>
RGB 동영상과 알파 동영상을 공동으로 생성하기 위해 사전 학습된 RGB 동영상 생성 모델에 여러 수정을 적용한다. 

먼저, noise가 더해진 입력 토큰의 시퀀스 길이를 $$\textbf{x}_\textrm{video}^{1:L}$$에서 $$\textbf{x}_\textrm{video}^{1:2L}$$로 두 배로 늘려 모델이 두 배 길이의 동영상을 생성할 수 있도록 한다. 여기서 $$\textbf{x}_\textrm{video}^{1:L}$$은 RGB 비디오로 디코딩되고, $$\textbf{x}_\textrm{video}^{(L+1):2L}$$은 알파 동영상으로 디코딩된다. Query, key, value 표현은 다음과 같다.

$$
\begin{equation}
\{\textbf{Q}, \textbf{K}, \textbf{V}\} = [\textbf{W}_{\{q,k,v\}} (\textbf{x}_\textrm{text}); \textbf{f}_{\{q,k,v\}} (\textbf{x}_\textrm{video}^{1:2L})]
\end{equation}
$$

($\textbf{W}$는 transformer의 projection 행렬, $\textbf{f}$는 비주얼 토큰을 위한 projection과 positional encoding)

시퀀스를 두 배로 늘리는 것 외에도 batch size나 잠재 차원을 늘리고 출력을 두 개의 도메인으로 분할하는 방법도 살펴보았습니다. 그러나 이러한 접근 방식은 나중에 논의할 제한된 데이터 세트에서는 효과성이 제한적인 것으로 나타났습니다.

<center><img src='{{"/assets/img/transpixeler/transpixeler-fig4.webp" | relative_url}}' width="75%"></center>
<br>
또한, positional encoding 함수 $\textbf{f}$를 수정한다. 인덱스를 연속적으로 넘버링하는 대신 RGB 토큰과 알파 토큰이 동일한 positional encoding을 공유하도록 허용한다. 예를 들어, absolute positional encoding을 사용하는 경우 다음과 같다.

$$
\begin{equation}
\textbf{f}_{\{q,k,v\}}^\ast (\textbf{x}_\textrm{video}) = \begin{cases}
\textbf{W}_{\{q,k,v\}} (\textbf{x}_\textrm{video}^m + \textbf{p}^m) & \textrm{if} \; m \le L \\
\textbf{W}_{\{q,k,v\}}^\ast (\textbf{x}_\textrm{video}^m + \textbf{p}^{m-L} + d) & \textrm{if} \; m > L
\end{cases}
\end{equation}
$$

$d$는 0으로 초기화된 추가로 도입된 도메인 임베딩으로, RGB 토큰과 알파 토큰을 모델이 적응적으로 구별할 수 있도록 학습 가능하게 만든다. $d$를 사용하지 않으면 다른 noise로 초기화하더라도 두 도메인의 토큰이 동일한 결과를 생성하는 경향이 있다. 도메인 임베딩의 도입은 학습의 시작 부분에서 공간-시간 정렬 문제를 최소화하여 수렴을 가속화한다.

다음으로, 저자들은 [LoRA](https://kimjy99.github.io/논문리뷰/lora)를 사용하는 fine-tuning 방식을 제안하였다. 여기서 LoRA layer는 알파 토큰에만 적용된다.

$$
\begin{equation}
\textbf{W}_{\{q,k,v\}}^\ast (\textbf{x}_\textrm{video}^m + \textbf{p}^{m-L} + d) = \textbf{W}_{\{q,k,v\}} (\textbf{x}_\textrm{video}^m + \textbf{p}^{m-L} + d) + \gamma \cdot \textrm{LoRA} (\textbf{x}_\textrm{video}^m + \textbf{p}^{m-L} + d)
\end{equation}
$$

또한, 저자들은 원치 않는 attention 계산을 차단하기 위해 attention mask를 설계하였다. 텍스트-동영상 토큰 시퀀스의 길이가 $L_\textrm{text} + 2L$이면, 마스크는 다음과 같이 정의된다.

$$
\begin{equation}
\textbf{M}_{mn}^\ast = \begin{cases} -\infty & \textrm{if} \; m \le L_\textrm{text} \; \textrm{and} \; n > L_\textrm{text} + L \\ 0 & \textrm{otherwise} \end{cases}
\end{equation}
$$

이러한 수정 사항들을 모두 결합하면, 전체 inference는 다음과 같이 표현된다.

$$
\begin{equation}
\textrm{Attention} (\textbf{Q}, \textbf{K}, \textbf{V}) = \textrm{softmax} \left( \frac{\textbf{Q}\textbf{K}^\top}{\sqrt{d_k}} + \textbf{M}^\ast \right) \textbf{V} \\
\textrm{where} \; \{\textbf{Q}, \textbf{K}, \textbf{V}\} = [\textbf{W}_{\{q,k,v\}} (\textbf{x}_\textrm{text}); \textbf{f}_{\{q,k,v\}}^\ast (\textbf{x}_\textrm{video})]
\end{equation}
$$

학습은 flow matching 또는 diffusion process를 사용하여 수행된다.

#### Analysis
<center><img src='{{"/assets/img/transpixeler/transpixeler-fig5.webp" | relative_url}}' width="80%"></center>
<br>
사전 학습된 동영상 모델의 능력을 극대화하여 기존 RGBA 데이터셋을 넘어서 생성할 수 있도록 하기 위해, 저자들은 3D full attention DiT 동영상 생성 모델 내에서 가장 중요한 구성 요소인 attention 메커니즘을 분석하였다. Attention 행렬 $\textbf{Q} \textbf{K}^\top$는 차원이 $(L_\textrm{text} + 2L) \times (L_\textrm{text} + 2L)$이며, 단순하게 3$\times$3 grouped attention 행렬로 볼 수 있다.

##### Text-Attend-to-RGB & RGB-Attend-to-Text
두 attention 메커니즘은 왼쪽 위 2$\times$2 섹션을 나타내며 원래 RGB 생성 모델에 존재하는 계산이다. 이 계산 부분이 영향을 받지 않도록 하면 원래 RGB 생성 성능을 복제할 수 있다. 따라서 LoRA의 영향 범위를 제한하여 텍스트와 RGB 토큰 모두에 대한 원래 QKV 값을 유지하고 두 도메인에 대한 사전 학습된 모델의 동작을 보존한다. 

LoRA 외에도, 텍스트 토큰과 RGB 토큰은 query와​ key로 알파 토큰과 상호 작용해야 하므로 이 2$\times$2 attention 행렬의 계산이 변경됩니다. 따라서, 저자들은 RGB 생성에 영향을 미치는 두 가지 추가 attention 계산을 추가로 분석하였다.

##### Text-Attend-to-Alpha
저자들은 Text-attend-to-Alpha가 생성 품질에 해롭다는 것을 발견했다. 모델은 원래 텍스트와 RGB 데이터로 학습되었기 때문에 텍스트에서 알파로의 attention를 도입하면 알파와 RGB 사이의 도메인 차이로 인해 간섭이 발생한다. 구체적으로, 알파는 윤곽 정보만 제공하고 텍스트 프롬프트와 관련된 풍부한 텍스처, 색상, semantic 디테일이 부족하여 생성 품질이 저하된다. 이를 완화하기 위해, 앞서 설명한 대로 이 계산을 차단하는 attention mask를 사용한다.

##### RGB-Attend-to-Alpha
대조적으로, RGB-attend-to-Alpha는 성공적인 공동 생성에 필수적이다. 모델은 이 attention을 통해 알파 정보를 고려하여 RGB 토큰을 정제하고 생성된 RGB와 알파 채널 간의 정렬을 용이하게 할 수 있다. 이 정제 프로세스는 알파 guidance에 기반한 RGB 정제를 위한 피드백 메커니즘이 부족했던 이전의 생성 후 예측 파이프라인에서 누락된 구성 요소이다. 

## Experiments
- 데이터셋: VideoMatte240K
- 구현 디테일
  - base model
    - [CogVideoX](https://kimjy99.github.io/논문리뷰/cogvideox) (480$\times$720, 49 프레임, 8 FPS)
    - CogVideoX의 수정 버전 (176$\times$320, 64 프레임, 24 FPS)
    - 샘플링 step 수: 50
  - LoRA rank: 128
  - iteration: 5,000
  - batch size: 8
  - GPU: NVIDIA A100 8개

### 1. Applications
다음은 (위) text-to-video와 (아래) image-to-video의 예시들이다. 

<center><img src='{{"/assets/img/transpixeler/transpixeler-fig6.webp" | relative_url}}' width="100%"></center>

### 2. Comparisons
다음은 생성 후 예측 파이프라인과 공동 생성 파이프라인을 비교한 예시이다. 

<center><img src='{{"/assets/img/transpixeler/transpixeler-fig2.webp" | relative_url}}' width="75%"></center>
<br>
다음은 생성 후 예측 파이프라인과 비교한 결과이다. 

<center><img src='{{"/assets/img/transpixeler/transpixeler-fig7.webp" | relative_url}}' width="100%"></center>
<br>
다음은 공동 생성 파이프라인과 비교한 결과이다. (위: LayerDiffusion + AnimateDiff, 아래: TransPixeler)

<center><img src='{{"/assets/img/transpixeler/transpixeler-fig8.webp" | relative_url}}' width="100%"></center>
<br>
다음은 user study 결과이다. 

<center><img src='{{"/assets/img/transpixeler/transpixeler-table1.webp" | relative_url}}' width="55%"></center>

### 3. Ablation Study
다음은 공동 생성을 위한 대체 디자인들이다. (b)가 본 논문의 방법이다. 

<center><img src='{{"/assets/img/transpixeler/transpixeler-fig9.webp" | relative_url}}' width="60%"></center>
<br>
다음은 ablation 결과이다. ((a) 본 논문의 방법, (b) RGB-attend-to-Alpha 제외, (c) Text-attend-to-Alpha 추가, (d) Batch Extension, (e) Latent Dimension Extension)

<center><img src='{{"/assets/img/transpixeler/transpixeler-fig10.webp" | relative_url}}' width="75%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/transpixeler/transpixeler-fig11.webp" | relative_url}}' width="45%"></center>