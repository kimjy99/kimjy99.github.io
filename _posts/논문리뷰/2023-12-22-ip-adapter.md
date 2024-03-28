---
title: "[논문리뷰] IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models"
last_modified_at: 2023-12-22
categories:
  - 논문리뷰
tags:
  - Diffusion
  - Text-to-Image
  - Computer Vision
  - AI
excerpt: "IP-Adapter 논문 리뷰"
use_math: true
classes: wide
---

> arXiv 2023. [[Paper](https://arxiv.org/abs/2308.06721)] [[Page](https://ip-adapter.github.io/)] [[Github](https://github.com/tencent-ailab/IP-Adapter)]  
> Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, Wei Yang  
> Tencent AI Lab  
> 13 Aug 2023  

<center><img src='{{"/assets/img/ip-adapter/ip-adapter-fig1.PNG" | relative_url}}' width="100%"></center>

## Introduction
이미지 생성은 최근 대규모 text-to-image (T2I) diffusion model의 성공으로 놀라운 발전을 이루었다. 사용자는 강력한 T2I diffusion model을 사용하여 이미지를 생성하는 텍스트 프롬프트를 작성할 수 있다. 그러나 원하는 콘텐츠를 생성하기 위해 좋은 텍스트 프롬프트를 작성하는 것은 복잡한 프롬프트 엔지니어링이 필요한 경우가 많기 때문에 쉽지 않다. 또한, 텍스트는 복잡한 장면이나 개념을 표현하는 데에는 유익하지 않아 콘텐츠 제작에 방해가 될 수 있다. 

이와 같은 텍스트 프롬프트 제한 사항을 고려하여 이미지를 생성할 수 있는 다른 프롬프트 유형이 있을까? 자연스러운 선택은 이미지 프롬프트를 사용하는 것이다. 왜냐하면 이미지는 텍스트에 비해 더 많은 내용과 디테일을 표현할 수 있기 때문이다. DALL-E 2는 이미지 프롬프트를 지원하기 위한 첫 번째 시도이며, diffusion model은 텍스트 임베딩이 아닌 이미지 임베딩으로 컨디셔닝되며, T2I 능력을 달성하려면 prior 모델이 필요하다. 그러나 대부분의 기존 T2I diffusion model은 이미지를 생성하기 위해 텍스트로 컨디셔닝된다. 예를 들어 Stable Diffusion 모델은 고정된 CLIP 텍스트 인코더에서 추출된 텍스트 feature로 컨디셔닝된다. 이러한 T2I diffusion model에서도 이미지 프롬프트가 지원될 수 있을까? 본 논문은 간단한 방식으로 이러한 T2I diffusion model에 대한 이미지 프롬프트를 통해 생성 능력을 활성화하려고 시도하였다.

SD Image Variations와 Stable unCLIP과 같은 이전 연구들에서는 이미지 프롬프팅 능력을 달성하기 위해 이미지 임베딩에서 직접 텍스트 조건부 diffusion model을 fine-tuning하였으며 그 효과가 입증되었다. 그러나 이 접근법의 단점은 명백하다. 

1. 텍스트를 사용하여 이미지를 생성하는 원래의 능력을 제거하고 fine-tuning을 위해 대규모 컴퓨팅 리소스가 필요한 경우가 많다. 
2. Fine-tuning된 모델은 일반적으로 재사용이 불가능하다. 이미지 프롬프팅 능력을 동일한 T2I 기반 모델에서 파생된 다른 커스텀 모델로 직접 전송할 수 없기 때문이다. 
3. 새로운 모델은 [ControlNet](https://kimjy99.github.io/논문리뷰/controlnet)과 같은 기존 구조 제어 도구와 호환되지 않는 경우가 많아 다운스트림 애플리케이션에 심각한 문제를 야기한다. 

Fine-tuning의 단점으로 인해 일부 연구에서는 diffusion model의 fine-tuning을 피하면서 텍스트 인코더를 이미지 인코더로 대체하는 것을 선택하였다. 이 방법은 효과적이고 간단하지만 여전히 몇 가지 단점이 있다. 

1. 이미지 프롬프트만 지원되므로 사용자가 텍스트와 이미지 프롬프트를 동시에 사용하여 이미지를 생성할 수 없다. 
2. 이미지 인코더를 fine-tuning하는 것만으로는 이미지 품질을 보장하기에 충분하지 않은 경우가 많으며 일반화 문제가 발생할 수 있다.

저자들은 원본 T2I 모델을 수정하지 않고도 이미지 프롬프트를 사용할 수 있는지에 관심을 가졌다. ControlNet과 [T2I-adapter](https://kimjy99.github.io/논문리뷰/t2i-adapter)에서는 이미지 생성을 가이드하기 위해 기존 T2I diffusion model에 추가 네트워크를 효과적으로 연결할 수 있음이 입증되었다. 이를 위해 CLIP 이미지 인코더에서 추출된 이미지 feature는 학습 가능한 네트워크를 통해 새로운 feature에 매핑된 다음 텍스트 feature와 concatenate된다. 원본 텍스트 feature를 대체함으로써 병합된 feature가 diffusion model의 UNet에 공급되어 이미지 생성을 가이드한다. 이러한 어댑터는 이미지 프롬프트를 사용하는 방법으로 볼 수 있지만 생성된 이미지는 프롬프팅된 이미지에 부분적으로만 충실하며, 결과가 처음부터 학습된 모델은 물론 fine-tuning된 이미지 프롬프트 모델보다 더 나쁜 경우가 많다.

저자들은 앞서 언급한 방법의 주요 문제점이 T2I diffusion model의 cross-attention 모듈에 있다고 주장하였다. 사전 학습된 diffusion model에서 cross-attention 레이어의 key와 value projection 가중치는 텍스트 feature에 맞게 학습된다. 결과적으로, 이미지 feature와 텍스트 feature를 cross-attention 레이어에 병합하면 이미지 feature를 텍스트 feature에 정렬하는 것만 달성되며 이로 인해 잠재적으로 일부 이미지 관련 정보가 누락되어 결국 레퍼런스 이미지를 사용한 제어 가능한 대략적인 생성만 가능하게 된다. 

저자들은 이전 방법의 단점을 피하기 위해 **IP-Adapter**라는 보다 효과적인 이미지 프롬프트 어댑터를 제안하였다. 특히 IP-Adapter는 텍스트 feature와 이미지 feature에 대해 decoupled cross-attention 메커니즘을 채택하였다. UNet diffusion model의 모든 cross-attention 레이어에 대해 이미지 feature에 대해서만 추가 cross-attention 레이어를 추가한다. 학습 단계에서는 새로운 cross-attention 레이어의 파라미터만 학습되고 원래 UNet 모델은 그대로 유지된다. IP-Adapter는 가볍지만 매우 효율적이다. 2200만 개의 파라미터만을 가진 IP-Adapter의 생성 성능은 T2I diffusion model에서 완전히 fine-tuning된 이미지 프롬프트 모델과 비슷하다. 더 중요한 것은 IP-Adapter가 뛰어난 일반화 능력을 보여주고 텍스트 프롬프트와 호환된다는 것이다. 제안된 IP-Adapter를 사용하면 다양한 이미지 생성 작업을 쉽게 수행할 수 있다.

## Method
### 1. Image Prompt Adapter
<center><img src='{{"/assets/img/ip-adapter/ip-adapter-fig2.PNG" | relative_url}}' width="80%"></center>
<br>
본 논문에서 이미지 프롬프트 어댑터는 사전 학습된 T2I diffusion model을 사용하여 이미지 프롬프트가 포함된 이미지를 생성할 수 있도록 디자인되었다. 현재 어댑터들은 fine-tuning된 이미지 프롬프트 모델이나 처음부터 학습된 모델의 성능을 맞추는 데 어려움을 겪고 있다. 가장 큰 이유는 이미지 feature를 사전 학습된 모델에 효과적으로 임베딩할 수 없기 때문이다. 대부분의 방법은 단순히 concatenate된 feature를 고정된 cross-attention 레이어에 공급하여 diffusion model이 이미지 프롬프트에서 세밀한 feature를 캡처하는 것을 막는다. 이 문제를 해결하기 위해 저자들은 새로 추가된 cross-attention 레이어에 이미지 feature가 포함되는 decoupled cross-attention 전략을 제시하였다. 제안된 IP-Adapter의 전체 아키텍처는 위 그림에 나와 있다. IP-Adapter는 두 부분으로 구성된다. 

1. 이미지 프롬프트에서 이미지 feature를 추출하는 이미지 인코더
2. 이미지 feature를 사전 학습된 T2I diffusion model에 삽입하기 위해 decoupled cross-attention이 있는 적응형 모듈 

#### Image Encoder
대부분의 방법들을 따라 사전 학습된 CLIP 이미지 인코더 모델을 사용하여 이미지 프롬프트에서 이미지 feature를 추출한다. CLIP 모델은 이미지-텍스트 쌍이 포함된 대규모 데이터셋에 대한 contrastive learning을 통해 학습된 멀티모달 모델이다. 이미지 캡션과 잘 정렬되고 이미지의 풍부한 콘텐츠와 스타일을 표현할 수 있는 CLIP 이미지 인코더의 글로벌 이미지 임베딩을 활용한다. 학습 단계에서는 CLIP 이미지 인코더가 고정된다.

글로벌 이미지 임베딩을 효과적으로 분해하기 위해 학습 가능한 작은 projection network를 사용하여 이미지 임베딩을 길이 $N$ (본 논문에서는 $N = 4$ 사용)의 feature 시퀀스로 project한다. 이미지 feature의 차원은 사전 학습된 diffusion model의 텍스트 feature 차원과 동일하다. 본 논문에서 사용한 projection network는 linear layer와 Layer Normalization으로 구성된다.

#### Decoupled Cross-Attention
이미지 feature는 cross-attention이 분리된 적응형 모듈을 통해 사전 학습된 UNet 모델에 통합된다. 원본 Stable Diffusion 모델에서 CLIP 텍스트 인코더의 텍스트 feature는 cross-attention 레이어에 공급되어 UNet 모델에 연결된다. Query feature $Z$와 텍스트 feature $c_t$가 주어지면 cross-attention의 출력 $Z^\prime$은 다음 방정식으로 정의될 수 있다.

$$
\begin{equation}
Z^\prime = \textrm{Attention} (Q, K, V) = \textrm{Softmax} (\frac{QK^\top}{\sqrt{d}}) V \\
\textrm{where} \; Q = ZW_q, \; K = c_t W_k, \; V = c_t W_v
\end{equation}
$$

$Q$, $K$, $V$는 attention 연산의 query, key, value 행렬이고, $W_q$, $W_k$, $W_v$는 학습 가능한 linear projection layer의 가중치 행렬이다. 

저자들은 텍스트 feature와 이미지 특징에 대한 cross-attention 레이어가 분리된 decoupled cross-attention 메커니즘을 제안하였다. 구체적으로 말하면 원본 UNet 모델의 각 cross-attention 레이어에 대해 새로운 cross-attention 레이어를 추가하여 이미지 feature를 삽입한다. 이미지 feature $c_i$가 주어지면 새로운 cross-attention의 출력 $Z^{\prime \prime}$은 다음과 같이 계산된다.

$$
\begin{equation}
Z^{\prime \prime} = \textrm{Attention} (Q, K^\prime, V^\prime) = \textrm{Softmax} (\frac{Q(K^\prime)^\top}{\sqrt{d}}) V^\prime \\
\textrm{where} \; Q = ZW_q, K^\prime = c_i W_k^\prime, V^\prime = c_i W_v^\prime
\end{equation}
$$

텍스트 cross-attention과 이미지 cross-attention에 대해 동일한 query를 사용한다. 결과적으로, 각 cross-attention 레이어에 대해 두 개의 파라미터 $W_k^\prime$, $W_v^\prime$만 추가하면 된다. 수렴 속도를 높이기 위해 $W_k^\prime$과 $W_v^\prime$은 $W_k$와 $W_v$에서 초기화된다. 그런 다음 이미지 cross-attention 출력을 텍스트 cross-attention 출력에 더하기만 하면 된다. 따라서 decoupled cross-attention의 최종 공식은 다음과 같이 정의된다.

$$
\begin{equation}
Z^\textrm{new} = \textrm{Softmax} (\frac{QK^\top}{\sqrt{d}}) V + \textrm{Softmax} (\frac{Q(K^\prime)^\top}{\sqrt{d}}) V^\prime
\end{equation}
$$

원래 UNet 모델을 고정했으므로 $W_k^\prime$과 $W_v^\prime$만 학습 가능하다.

#### Training and Inference
학습 중에는 사전 학습된 diffusion model의 파라미터를 고정된 상태로 유지하면서 IP-Adapter만 최적화한다. 또한 IP-Adapter는 원본 Stable Diffusion과 동일한 학습 목적 함수를 사용하여 이미지-텍스트 쌍이 포함된 데이터셋에 대해 학습된다.

$$
\begin{equation}
L_\textrm{simple} = \mathbb{E}_{x_0, \epsilon, c_t, c_i, t} \| \epsilon - \epsilon_\theta (x_t, c_t, c_i, t) \|^2
\end{equation}
$$

또한 inference 단계에서 classifier-free guidance를 활성화하기 위해 학습 단계에서 이미지 조건을 무작위로 제거한다. 

$$
\begin{equation}
\hat{\epsilon}_\theta (x_t, c_t, c_i, t) = w \epsilon_\theta (x_t, c_t, c_i, t) + (1-w) \epsilon_\theta (x_t, t)
\end{equation}
$$

이미지 조건이 제거되면 CLIP 이미지 임베딩을 0으로 설정한다.

텍스트 cross-attention과 이미지 cross-attention이 분리되어 있으므로 inference 단계에서 이미지 조건의 가중치를 조정할 수도 있다. 

$$
\begin{equation}
Z^\textrm{new} = \textrm{Attention} (Q, K, V) + \lambda \cdot \textrm{Attention} (Q, K^\prime, V^\prime)
\end{equation}
$$

여기서 $\lambda$는 가중치이고 $\lambda = 0$인 경우 모델은 원본 T2I diffusion model이 된다. 

## Experiment
- 데이터셋: LAION-2B, COYO-700
- 구현 디테일
  - Diffusion model: Stable Diffusion v1.5
  - 이미지 인코더: OpenCLIP ViT-H/14
  - Stable Diffusion에 16개의 cross-attention 레이어가 있으므로 각 레이어마다 새로운 이미지 cross-attention 레이어를 추가
  - 빠른 학습을 위해 DeepSpeed ZeRO-2를 사용
  - 8개의 V100 GPU에서 학습
  - step: 100만
  - batch size: GPU당 8
  - optimizer: AdamW
  - learning rate: 0.0001 (고정)
  - weight decay: 0.01
  - 이미지 해상도: 512$\times$512
  - classifier-free guidance
    - 이미지 조건만 제거: 0.05
    - 텍스트 조건만 제거: 0.05
    - 둘 다 동시에 제거: 0.05
    - guidance scale: 7.5
  - $\lambda$ = 1.0

### 1. Comparison with Existing Methods
다음은 COCO validation set에서 다른 방법들과 IP-Adapter를 비교한 표이다. 

<center><img src='{{"/assets/img/ip-adapter/ip-adapter-table1.PNG" | relative_url}}' width="100%"></center>
<br>
다음은 여러 종류와 스타일의 이미지를 조건으로 다른 방법들과 IP-Adapter를 비교한 결과이다. 

<center><img src='{{"/assets/img/ip-adapter/ip-adapter-fig3.PNG" | relative_url}}' width="100%"></center>

### 2. More Results
#### Generalizable to Custom Models
다음은 여러 diffusion model들과 IP-Adapter의 결과를 비교한 것이다. 

<center><img src='{{"/assets/img/ip-adapter/ip-adapter-fig4.PNG" | relative_url}}' width="100%"></center>

#### Structure Control
다음은 이미지 프롬프트와 추가적인 구조적 조건들에 대한 IP-Adapter의 결과들이다. (추가로 fine-tuning하지 않음)

<center><img src='{{"/assets/img/ip-adapter/ip-adapter-fig5.PNG" | relative_url}}' width="100%"></center>
<br>
다음은 여러 구조적 조건들에 대하여 다른 방법들과 결과를 비교한 것이다. 

<center><img src='{{"/assets/img/ip-adapter/ip-adapter-fig6.PNG" | relative_url}}' width="100%"></center>

#### Image-to-Image and Inpainting
다음은 IP-Adapter를 사용한 image-to-image와 인페인팅 결과이다. 

<center><img src='{{"/assets/img/ip-adapter/ip-adapter-fig7.PNG" | relative_url}}' width="90%"></center>

#### Multimodal Prompts
다음은 멀티모달 프롬프트에 대한 IP-Adapter의 결과들이다. 

<center><img src='{{"/assets/img/ip-adapter/ip-adapter-fig8.PNG" | relative_url}}' width="100%"></center>
<br>
다음은 멀티모달 프롬프트에 대하여 다른 방법들과 비교한 것이다. 

<center><img src='{{"/assets/img/ip-adapter/ip-adapter-fig9.PNG" | relative_url}}' width="100%"></center>

### 3. Ablation Study
다음은 간단한 어댑터와 결과를 비교한 것이다. 

<center><img src='{{"/assets/img/ip-adapter/ip-adapter-fig10.PNG" | relative_url}}' width="100%"></center>
<br>
다음은 글로벌한 feature를 사용하였을 떄와 세밀한 feature를 사용하였을 떄의 결과를 비교한 것이다. 

<center><img src='{{"/assets/img/ip-adapter/ip-adapter-fig11.PNG" | relative_url}}' width="100%"></center>