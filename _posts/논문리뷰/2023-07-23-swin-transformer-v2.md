---
title: "[논문리뷰] Swin Transformer V2: Scaling Up Capacity and Resolution"
last_modified_at: 2023-07-23
categories:
  - 논문리뷰
tags:
  - ViT
  - Image Classification
  - Object Detection
  - Image Segmentation
  - Computer Vision
  - AI
  - Microsoft
  - CVPR
excerpt: "Swin Transformer V2 논문 리뷰 (CVPR 2022)"
use_math: true
classes: wide
---

> CVPR 2022. [[Paper](https://arxiv.org/abs/2111.09883)] [[Github](https://github.com/microsoft/Swin-Transformer)]  
> Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, Baining Guo  
> Microsoft Research Asia  
> 18 Nov 2021  

## Introduction
언어 모델의 확장은 매우 성공적이었다. 언어 task에 대한 모델의 성능을 크게 향상시키고 모델은 인간과 유사한 놀라운 few-shot 능력을 보여주었다. 3억 4천만 개의 파라미터가 있는 BERT 대형 모델 이후 언어 모델은 몇 년 만에 1,000배 이상 빠르게 확장되어 5,300억 개의 dense parameter와 1조 6천억 개의 sparse parameter에 도달하였다. 또한 이러한 대규모 언어 모델은 광범위한 언어 task에 대해 인간 지능과 유사한 점점 더 강력한 few-shot 능력을 보유하고 있는 것으로 밝혀졌다.

반면에 비전 모델의 확장은 뒤처져 있다. 더 큰 비전 모델이 일반적으로 비전 task에서 더 잘 수행된다는 것이 오랫동안 인식되어 왔지만 절대적인 모델 크기는 최근에 약 10억에서 20억 개의 파라미터에 도달할 수 있었다. 더 중요한 것은 대형 언어 모델과 달리 기존의 대형 비전 모델은 이미지 분류 task에만 적용된다는 것이다.

크고 일반적인 비전 모델을 성공적으로 학습시키려면 몇 가지 주요 문제를 해결해야 한다. 첫째, 대형 비전 모델을 사용한 실험에서 학습의 불안정성 문제가 드러났다. 대형 모델에서 레이어 간 activation 진폭의 불일치가 훨씬 더 커짐을 발견했다. 원래 아키텍처를 자세히 살펴보면 이것이 기본 분기에 직접 다시 추가된 나머지 unit의 출력으로 인해 발생했음을 알 수 있다. 그 결과 activation 값이 레이어별로 누적되므로 더 깊은 레이어의 진폭이 초기 레이어의 진폭보다 훨씬 더 크다. 

본 논문은 이 문제를 해결하기 위해 res-post-norm이라는 새로운 정규화 구성을 제안한다. 이 구성은 LN 레이어를 각 residual unit의 시작 부분에서 뒷 부분으로 이동한다. 이 새로운 구성은 네트워크 레이어 전체에서 훨씬 더 온화한 activation 값을 생성한다. 또한 저자들은 이전의 내적 attention을 대체하기 위해 scaled cosine attention을 제안한다. Scaled cosine attention은 계산을 블록 입력의 진폭과 무관하게 만들고 attention 값이 극단으로 떨어질 가능성이 적다. 제안된 두 가지 테크닉은 학습 과정을 보다 안정적으로 만들 뿐만 아니라 특히 더 큰 모델의 정확도를 향상시킨다.

둘째, object detection과 semantic segmentation과 같은 많은 다운스트림 비전 task에는 고해상도 입력 이미지 또는 큰 attention window가 필요하다. 저해상도 사전 학습과 고해상도 fine-tuning 간의 window 크기 변화는 상당히 클 수 있다. 현재 일반적인 관행은 position bias map의 bi-cubic interpolation을 수행하는 것이다. 이 간단한 수정은 다소 임시방편이며 결과는 일반적으로 최선이 아니다. 로그 간격 좌표 입력에 작은 메타 네트워크를 적용하여 임의의 좌표 범위에 대한 바이어스 값을 생성하는 log-spaced continuous position bias (Log-CPB)를 도입한다. 메타 네트워크는 모든 좌표를 사용하므로 사전 학습된 모델은 메타 네트워크의 가중치를 공유하여 window 크기 간에 자유롭게 전송할 수 있다. 이 접근 방식의 중요한 설계는 타겟 window 크기가 사전 학습보다 훨씬 큰 경우에도 extrapolation (외삽) 비율이 낮을 수 있도록 좌표를 로그 공간으로 변환하는 것이다. 모델 용량과 해상도의 확장으로 인해 기존 비전 모델에서 엄청나게 높은 GPU 메모리 소비가 발생한다. 메모리 문제를 해결하기 위해 zero-optimizer, activation check pointing, sequential self-attention 계산의 새로운 구현을 포함한 몇 가지 중요한 기술을 통합한다. 이러한 기술을 사용하면 대형 모델과 해상도의 GPU 메모리 소비가 크게 줄어들고 학습 속도에는 미미한 영향만 미친다.


## Swin Transformer V2
### 1. A Brief Review of Swin Transformer
Swin Transformer는 영역 레벨 object detection, 픽셀 레벨 semantic segmentation과 이미지 레벨 image classification과 같은 다양한 인식 task에서 강력한 성능을 달성한 범용 컴퓨터 비전 backbone이다. Swin Transformer의 주요 아이디어는 두 가지의 강점을 결합하는 계층적 구조, locality, translation 불변성을 포함하여 Transformer 인코더에 몇 가지 중요한 시각적 prior를 도입하는 것이다. 기본 Transformer에는 강력한 모델링 능력이 있으며 시각적 prior는 다양한 비전 task에 친숙하다.

#### Normalization configuration
정규화 기술은 더 깊은 아키텍처를 안정적으로 학습시키는 데 중요하다. 원래 Swin Transformer는 광범위한 연구 없이 pre-normalization 구성을 활용하기 위해 언어 transformer와 ViT의 일반적인 관행을 계승한다. 

#### Relative position bias
Relative position bias는 self-attention 계산에서 기하학적 관계를 인코딩하기 위해 추가 바이어스 항을 도입하는 Swin Transformer의 핵심 구성 요소이다.

$$
\begin{equation}
\textrm{Attention} (Q, K, V) = \textrm{SoftMax} (\frac{QK^\top}{\sqrt{d}} + B) V
\end{equation}
$$

여기서 $B \in \mathbb{R}^{M^2 \times M^2}$는 각 head에 대한 relative position bias (RPB) 항이다. $Q, K, V \in \mathbb{R}^{M^2 \times d}$는 query, key, value 행렬이다. $d$는 query와 key의 차원이고 $M^2$는 window의 패치 수이다. Relative position bias는 시각적 요소의 상대적인 공간 구성을 인코딩하며 다양한 비전 task, 특히 object detection과 같은 dense recognition task에서 중요하다.

Swin Transformer에서 각 축을 따른 상대적 위치는 $[-M+1, M-1]$ 범위 내에 있고 RPB는 바이어스 행렬 $\hat{B} \in \mathbb{R}^{(2M-1) \times (2M-1)}$로 parameterize되며 $B$의 요소는 $\hat{B}$에서 가져온다. 서로 다른 window 크기에 걸쳐 전송할 때 사전 학습에서 학습된 RPB 행렬은 bi-cubic interpolation에 의한 fine-tuning에서 다른 크기의 바이어스 행렬을 초기화하는 데 사용된다.

#### Issues in scaling up model capacity and window resolution
저자들은 Swin Transformer의 용량과 window 해상도를 확장할 때 두 가지 문제를 관찰하였다.

<center><img src='{{"/assets/img/swin-transformer-v2/swin-transformer-v2-fig3.PNG" | relative_url}}' width="50%"></center>

1. **모델 용량을 확장할 때 불안정성 문제**: Swin Transformer 모델의 크기를 확장하면 더 깊은 레이어의 activation 값이 크게 증가한다. 가장 높은 진폭과 가장 낮은 진폭을 가진 레이어 간의 불일치는 $10^4$에 도달했다. 거대한 크기 (6.58억 파라미터)로 더 확장하면 위 그림에서 볼 수 있듯이 학습을 완료할 수 없다.
2. **Window 해상도 간에 모델을 전송할 때 성능이 저하됨**: Bi-cubic interpolation 접근법을 사용할 때, 사전 학습된 ImageNet-1K 모델 (256$\times$256 이미지, 8$\times$8 window 크기)의 정확도를 더 큰 이미지 해상도와 window 크기에서 직접 테스트할 때 정확도가 크게 감소한다. 

### 2. Scaling Up Model Capacity
앞서 언급했듯이 Swin Transformer와 대부분의 ViT는 바닐라 ViT에서 상속된 각 블록의 시작 부분에 layer norm (LN) 레이어를 채택한다. 모델 용량을 확장하면 더 깊은 레이어에서 activation 값의 상당한 증가가 관찰된다. 실제로 pre-normalization 구성에서 각 residual block의 출력 activation 값은 다시 기본 분기로 다시 병합되고 기본 분기의 진폭은 더 깊은 레이어에서 점점 더 커진다. 서로 다른 레이어의 큰 진폭 불일치로 인해 학습이 불안정해진다. 

#### Post normalization
<center><img src='{{"/assets/img/swin-transformer-v2/swin-transformer-v2-fig1.PNG" | relative_url}}' width="55%"></center>
<br>
본 논문은 이 문제를 완화하기 위해 위 그림과 같이 residual post normalization 접근 방식을 대신 사용할 것을 제안한다. 이 접근 방식에서 각 residual block의 출력은 메인 분기로 다시 병합되기 전에 정규화되며 메인 분기의 진폭은 레이어가 더 깊어지면 축적되지 않는다. 이 접근 방식에 의한 activation 진폭은 원래 pre-normalization 구성보다 훨씬 약하다. 가장 큰 모델의 학습에서는 학습을 더욱 안정화하기 위해 6개의 Transformer 블록마다 기본 분기에 추가 LN 레이어를 도입한다.

#### Scaled cosine attention
원래 self-attention 계산에서 픽셀 쌍의 유사도 항은 query와 key 벡터의 내적으로 계산된다. 저자들은 이 접근법이 대규모 비전 모델에서 사용될 때 일부 블록과 head의 학습된 attention map이 특히 res-post-norm 구성에서 몇 개의 픽셀 쌍에 의해 지배되는 경우가 많다는 것을 발견했다. 본 논문은 이 문제를 완화하기 위해 스케일링된 코사인 함수로 픽셀 쌍 $i$와 $j$의 attention logit을 계산하는 scaled cosine attention 접근법을 제안하였다.

$$
\begin{equation}
\textrm{Sim} (q_i, k_j) = \frac{\cos (q_i, k_j)}{\tau} + B_{ij}
\end{equation}
$$

여기서 $B_{ij}$는 $i$와 $j$ 사이의 RPB이다. $\tau$는 학습 가능한 스칼라이며 head와 레이어 사이에 공유되지 않는다. $\tau$는 0.01보다 크게 설정된다. 코사인 함수는 이미 정규화되어 있으므로 온화한 attention 값을 얻을 수 있다. 

### 3. Scaling Up Window Resolution
#### Continuous relative position bias
Parameterize된 바이어스를 직접 최적화하는 대신 continuous position bias (CPB) 접근 방식은 상대 좌표에서 작은 메타 네트워크를 채택한다. 

$$
\begin{equation}
B (\Delta x, \Delta y) = \mathcal{G} (\Delta x, \Delta y)
\end{equation}
$$

여기서 $\mathcal{G}$는 작은 네트워크, 예를 들어 ReLU가 사이에 있는 2-layer MLP이다. 

메타 네트워크 $\mathcal{G}$는 임의의 상대 좌표에 대한 바이어스 값을 생성하므로 window 크기를 임의로 변경하는 fine-tuning task로 자연스럽게 전환될 수 있다. Inference에서 각 상대 위치의 바이어스 값은 미리 계산되어 모델 파라미터로 저장될 수 있으므로 inference는 원래 parameterize된 바이어스 접근법과 동일하다.

#### Log-spaced coordinates
매우 다양한 window 크기에 걸쳐 전송할 때 상대 좌표 범위의 많은 부분을 외삽해야 한다. 이 문제를 완화하기 위해 저자들은 원래 선형 간격 좌표 대신 로그 간격 좌표를 사용할 것을 제안하였다.

$$
\begin{equation}
\hat{\Delta x} = \textrm{sign} (x) \cdot \log (1 + \vert \Delta x \vert) \\
\hat{\Delta y} = \textrm{sign} (y) \cdot \log (1 + \vert \Delta y \vert)
\end{equation}
$$

여기서 $\Delta x$, $\Delta y$는 선형 간격 좌표이고 $\hat{\Delta x}$, $\hat{\Delta y}$는 로그 간격 좌표이다.

로그 간격 좌표를 사용하면 window 해상도에 걸쳐 RPB를 전송할 때 필요한 외삽 비율이 원래 선형 간격 좌표를 사용하는 것보다 훨씬 작아진다. 예를 들어, 원래 좌표를 사용하여 사전 학습된 8$\times$8 window 크기에서 fine-tuning된 16$\times$16 window 크기로 전송하는 경우 입력 좌표 범위는 $[-7, 7] \times [-7, 7]$에서 $[-15, 15] \times [-15, 15]$이다. 외삽 비율은 원래 범위의 $8/7 = 1.14 \times$이다. 로그 간격 좌표를 사용하면 입력 범위는 $[-2.079, 2.079] \times [-2.079 \times 2.079]$에서 $[-2.733, 2.733] \times [-2.733 \times 2.733]$이다. 외삽 비율은 원래 범위의 0.33배로 원래 선형 간격 좌표를 사용하는 것보다 약 4배 작은 외삽 비율이다.

### 4. Self-Supervised Pre-training
더 큰 모델은 더 많은 데이터를 필요로 한다. 데이터 부족 문제를 해결하기 위해 이전의 대형 비전 모델은 일반적으로 JFT-3B와 같은 대용량 레이블 데이터를 사용하였다. 본 논문에서는 레이블이 지정된 데이터에 대한 요구를 완화하기 위해 self-supervised 사전 학습 방법인 SimMIM을 활용한다. 저자들은 이 접근 방식을 통해 7천만 개의 레이블이 지정된 이미지만 사용하여 4개의 대표적인 비전 벤치마크에서 SOTA를 달성하는 30억 개의 파라미터의 강력한 Swin Transformer 모델을 성공적으로 학습시켰다.

### 5. Implementation to Save GPU Memory
또 다른 문제는 용량과 해상도가 모두 큰 경우 일반 구현으로 감당할 수 없는 GPU 메모리 소비에 있다. 메모리 문제를 해결하기 위해 다음 구현을 채택한다.

1. **Zero-Redundancy Optimizer (ZeRO)**: Optimizer의 일반적인 데이터 병렬 구현에서 모델 파라미터와 최적화 상태는 모든 GPU에 브로드캐스트된다. 이 구현은 GPU 메모리 소비에 매우 우호적이지 않다. 예를 들어, 30억 개의 파라미터 모델은 AdamW 옵티마이저와 fp32 가중치/상태가 사용될 때 48GB의 GPU 메모리를 소비한다. ZeRO optimizer를 사용하면 모델 파라미터와 해당 최적화 상태가 분할되어 여러 GPU에 분산되어 메모리 소비가 크게 줄어든다. 저자들은 DeepSpeed 프레임워크를 채택하고 ZeRO 1단계 옵션을 사용하였다. 이 최적화는 학습 속도에 거의 영향을 미치지 않는다.
2. **Activation check-pointing**: Transformer 레이어의 feature map도 많은 GPU 메모리를 사용하므로 이미지 및 window 해상도가 높을 때 병목 현상이 발생할 수 있다. Activation check-pointing 기술은 메모리 소비를 크게 줄일 수 있는 반면 학습 속도는 최대 30% 느려진다.
3. **Sequential self-attention computation**: 예를 들어 window 크기가 32$\times$32인 1,536$\times$1,536 해상도의 이미지와 같이 매우 큰 해상도에서 대형 모델을 학습시키려면 일반 A100 GPU (40GB 메모리)는 위의 두 가지 최적화 기술을 사용하더라도 여전히 저렴하지 않다. 이 경우 self-attention 모듈이 병목 현상을 일으킨다. 이 문제를 완화하기 위해 이전의 일괄 계산 방식을 사용하는 대신 순차적으로 self-attention 계산을 구현한다. 이 최적화는 처음 두 단계의 레이어에 적용되며 전체 학습 속도에 거의 영향을 미치지 않는다.

### 6. Model configurations
Swin Transformer V2의 다음 4가지 구성에 대해 원래 Swin Transformer의 stage, 블록, 채널 설정을 유지한다. 

- **SwinV2-T**: $C$ = 96, 블럭 수 = {2, 2, 6, 2}
- **SwinV2-S**: $C$ = 96, 블럭 수 = {2, 2, 18, 2}
- **SwinV2-B**: $C$ = 128, 블럭 수 = {2, 2, 18, 2}
- **SwinV2-L**: $C$ = 192, 블럭 수 = {2, 2, 18, 2}

여기서 $C$는 첫 번째 stage의 채널 수이다. 

저자들은 여기에 Swin Transformer V2의 크기를 키운 2가지 구성인 huge와 giant를 추가하였다. 

- **SwinV2-H**: $C$ = 352, 블럭 수 = {2, 2, 18, 2}, 파라미터 6.58억 개
- **SwinV2-G**: $C$ = 512, 블럭 수 = {2, 2, 42, 4}, 파라미터 30억 개

SwinV2-H와 SwinV2-G의 경우 6개 레이어마다 기본 분기에 추가 LN 레이어를 추가하였다. 

## Experiments
- 데이터셋
  - ImageNet-1K image classification
  - COCO object detection
  - ADE20K semantic segmentation
  - Kinetics-400 video action recognition (SwinV2-G)

### 1. Scaling Up Experiments
다음은 ImageNet-1K V1/V2에서 대형 비전 모델들과 classification 결과를 비교한 표이다.

<center><img src='{{"/assets/img/swin-transformer-v2/swin-transformer-v2-table2.PNG" | relative_url}}' width="100%"></center>
<br>
다음은 COCO object detection과 instance segmentation에서 이전 최고의 결과들과 비교한 표이다. I(W)는 이미지와 window의 크기를 뜻한다.

<center><img src='{{"/assets/img/swin-transformer-v2/swin-transformer-v2-table3.PNG" | relative_url}}' width="48%"></center>
<br>
다음은 ADE20K semantic segmentation에서 이전 최고의 결과들과 비교한 표이다. 

<center><img src='{{"/assets/img/swin-transformer-v2/swin-transformer-v2-table4.PNG" | relative_url}}' width="45%"></center>
<br>
다음은 Kinetics-400 video action classification에서 이전 최고의 결과들과 비교한 표이다. 

<center><img src='{{"/assets/img/swin-transformer-v2/swin-transformer-v2-table5.PNG" | relative_url}}' width="55%"></center>

### 2. Ablation Study
다음은 res-post-norm과 cosine attention에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/swin-transformer-v2/swin-transformer-v2-table6.PNG" | relative_url}}' width="49%"></center>
<br>
다음은 다른 정규화 방법들과 비교한 표이다.

<center><img src='{{"/assets/img/swin-transformer-v2/swin-transformer-v2-table7.PNG" | relative_url}}' width="52%"></center>
<br>
다음은 다양한 모델 크기에 대한 신호 전파 그래프이다. 

<center><img src='{{"/assets/img/swin-transformer-v2/swin-transformer-v2-fig2.PNG" | relative_url}}' width="65%"></center>
<br>
다음은 다양한 position bias 접근법에 대한 비교 결과이다.

<center><img src='{{"/assets/img/swin-transformer-v2/swin-transformer-v2-table1.PNG" | relative_url}}' width="100%"></center>
<br>
다음은 다양한 모델 크기에서 Log-CPB에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/swin-transformer-v2/swin-transformer-v2-table8.PNG" | relative_url}}' width="49%"></center>