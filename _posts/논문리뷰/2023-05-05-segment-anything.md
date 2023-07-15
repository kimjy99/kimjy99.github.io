---
title: "[논문리뷰] Segment Anything (SAM)"
last_modified_at: 2023-05-05
categories:
  - 논문리뷰
tags:
  - Image Segmentation
  - ViT
  - Computer Vision
  - AI
  - Meta AI
excerpt: "SAM 논문 리뷰"
use_math: true
classes: wide
---

> arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02643)] [[Page](https://segment-anything.com/)] [[Github](https://github.com/facebookresearch/segment-anything)]  
> Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick  
> Meta AI Research, FAIR  
> 5 Apr 2023  

<center><img src='{{"/assets/img/segment-anything/segment-anything-fig1.PNG" | relative_url}}' width="100%"></center>

## Introduction
웹 규모 데이터셋에서 사전 학습된 대규모 언어 모델은 강력한 zero-shot 및 few-shot 일반화로 NLP를 혁신하고 있다. 이러한 "foundation model"은 학습 중에 볼 수 있는 것 이상으로 task와 데이터 분포를 일반화할 수 있다. 이 능력은 직접 만든 텍스트를 사용하여 언어 모델이 현재 task에 대한 유효한 텍스트 응답을 생성하도록 프롬프트하는 프롬프트 엔지니어링으로 구현되는 경우가 많다. 웹의 풍부한 텍스트 corpus로 확장하고 학습할 때 이러한 모델의 zero-shot 및 few-shot 성능은 fine-tuning된 모델과 놀라울 정도로 잘 비교된다. 경험적 추세는 모델 규모, 데이터셋 크기, total training compute로 이러한 동작이 개선됨을 보여준다.

그 정도는 적지만 컴퓨터 비전에서도 foundation model이 탐색되었다. 아마도 가장 눈에 띄는 예시는 쌍을 이루는 웹의 텍스트와 이미지를 align하는 것이다. 예를 들어 CLIP과 ALIGN은 contrastive learning을 사용하여 두 modality를 align하는 텍스트 및 이미지 인코더를 학습한다. 일단 학습되고 엔지니어링된 텍스트 프롬프트는 새로운 시각적 개념 및 데이터 분포에 대한 zero-shot 일반화가 가능하다. 이러한 인코더는 또한 다른 모듈과 효과적으로 구성되어 이미지 생성(ex. DALL·E)과 같은 하위 task를 가능하게 한다. 비전 및 언어 인코더에서 많은 진전이 있었지만 컴퓨터 비전에는 이 범위를 넘어서는 광범위한 문제가 포함되어 있으며 이러한 문제 중 많은 경우 풍부한 학습 데이터가 존재하지 않는다.

본 논문의 목표는 image segmentation을 위한 foundation model을 구축하는 것이다. 즉, 강력한 일반화를 가능하게 하는 task를 사용하여 광범위한 데이터셋에서 promptable model을 개발하고 사전 학습하는 것을 추구한다. 이 모델을 통해 프롬프트 엔지니어링을 사용하여 새로운 데이터 분포에 대한 다양한 하위 segmentation 문제를 해결하는 것을 목표로 한다.

이 계획의 성공은 세 가지 요소, task, 모델, 데이터에 달려 있다. 이를 개발하기 위해 image segmentation에 대한 다음 질문을 해결한다.

1. Zero-shot 일반화를 가능하게 하는 task는 무엇인가?
2. 해당 모델 아키텍처는 무엇인가?
3. 어떤 데이터가 이 task와 모델에 힘을 실어줄 수 있는가?

이러한 질문은 얽혀 있으며 포괄적인 해결책이 필요하다. 저자들은 강력한 사전 학습 목적 함수를 제공하고 광범위한 하위 애플리케이션을 활성화할 수 있을 만큼 충분히 일반적인 promptable segmentation task를 정의하는 것으로 시작한다. 이 task에는 유연한 프롬프트를 지원하고 상호 작용하도록 메시지가 표시될 때 segmentation mask를 실시간으로 출력할 수 있는 모델이 필요하다. 모델을 학습시키려면 다양하고 대규모의 데이터 소스가 필요하다. 안타깝게도 segmentation을 위한 웹 규모 데이터 소스가 없다. 이 문제를 해결하기 위해 "데이터 엔진"을 구축한다. 즉, 효율적인 모델을 사용하여 데이터 수집을 지원하는 것과 새로 수집된 데이터를 사용하여 모델을 개선하는 것 사이를 반복한다. 

## Segment Anything Task
<center><img src='{{"/assets/img/segment-anything/segment-anything-fig3.PNG" | relative_url}}' width="50%"></center>
<br>
저자들은 다음 토큰 예측 task가 foundation model 사전 학습에 사용되고 프롬프트 엔지니어링을 통해 다양한 하위 task를 해결하는 NLP에서 영감을 얻었다. Segmentation을 위한 foundation model을 구축하기 위해 유사한 능력을 가진 task를 정의하는 것을 목표로 한다.

#### Task
프롬프트의 아이디어를 NLP에서 segmentation으로 변환하는 것으로 시작한다. 여기에서 프롬프트는 전경/배경 점, 대략적인 박스 또는 마스크, 자유 형식 텍스트 또는 일반적으로 이미지에서 segmentation 대상을 나타내는 모든 정보가 될 수 있다. Promptable segmentation task는 프롬프트가 주어지면 유효한 segmentation mask를 반환하는 것이다. "유효한" 마스크의 요구 사항은 프롬프트가 모호하고 여러 개체를 참조할 수 있는 경우에 대해 출력이 그 객체 중 적어도 하나에 대한 합리적인 마스크여야 함을 의미한다. 이 요구 사항은 모호한 프롬프트에 대해 일관된 응답을 출력하는 언어 모델을 기대하는 것과 유사하다. 이 task를 선택하는 이유는 자연스러운 사전 학습 알고리즘과 프롬프팅을 통해 하위 segmentation task로의 zero-shot trasfer를 위한 일반적인 방법으로 이어지기 때문이다.

#### Pre-training
Promptable segmentation task는 각 학습 샘플에 대한 일련의 프롬프트(ex. 점, 박스, 마스크)를 시뮬레이션하고 모델의 마스크 예측을 ground-truth와 비교하는 자연스러운 사전 학습 알고리즘을 제안한다. 충분한 사용자 입력 후에 결국 유효한 마스크를 예측하는 것이 목표인 interactive segmentation과 달리 이 방법을 적용한다. 프롬프트가 모호한 경우에도 모든 프롬프트에 대해 항상 유효한 마스크를 예측하는 것이 목표이다. 이를 통해 데이터 엔진에서 요구하는 자동 주석을 포함하여 모호성이 포함된 use case에서 사전 학습된 모델이 효과적임을 보장한다. 저자들은 이 task를 잘 수행하는 것이 어렵고 전문적인 모델링 및 학습 loss 선택이 필요하다는 점에 주목하였다.

#### Zero-shot transfer
직관적으로 사전 학습 task는 모델이 inference 시간에 모든 프롬프트에 적절하게 응답할 수 있는 능력을 부여하므로 하위 task는 적절한 프롬프트를 엔지니어링하여 해결할 수 있다. 예를 들어 고양이에 대한 boundary box detector가 있는 경우 detector의 box 출력을 모델에 프롬프트로 제공하여 고양이 instance segmentation를 해결할 수 있다. 일반적으로 다양한 실용적인 segmentation task는 프롬프트로 캐스팅될 수 있다. 

#### Discussion
Prompting과 composition은 단일 모델을 확장 가능한 방식으로 사용하여 잠재적으로 모델 설계 시 알려지지 않은 task를 수행할 수 있도록 하는 강력한 도구이다. 이 접근 방식은 다른 foundation model이 사용되는 방식과 유사하다. 저자들은 프롬프트 엔지니어링과 같은 기술로 구동되는 composition 가능한 시스템 설계가 고정된 일련의 task를 위해 특별히 학습된 시스템보다 더 다양한 애플리케이션을 가능하게 할 것으로 예상하였다. Ccomposition의 관점에서 promptable segmentation과 interactive segmentation을 비교하는 것도 흥미롭다. interactive segmentation model은 인간 사용자를 염두에 두고 설계되었지만 promptable segmentation을 위해 학습된 모델은 더 큰 알고리즘 시스템으로 구성될 수 있다.

## Segment Anything Model
<center><img src='{{"/assets/img/segment-anything/segment-anything-fig4.PNG" | relative_url}}' width="100%"></center>
<br>
Segment Anything Model (SAM)에는 위 그림에 나와 있는 것처럼 세 가지 구성 요소가 있다. 

1. 이미지 인코더
2. 유연한 프롬프트 인코더
3. 빠른 마스크 디코더

저자들은 실시간 성능에 대한 특정 trade-off가 있는 Transformer 비전 모델을 구축하였다.

#### Image encoder
확장성과 강력한 사전 학습 방법에 동기를 부여받아 고해상도 입력을 처리하도록 최소한으로 조정된 MAE pre-trained ViT를 사용한다. 이미지 인코더는 이미지당 한 번 실행되며 모델을 프롬프트하기 전에 적용할 수 있다.

#### Prompt encoder
두 가지 집합의 프롬프트를 고려한다. 

1. Sparse(점, 박스, 텍스트)
2. Dense(마스크)

CLIP의 텍스트 인코더를 사용하여 각 프롬프트 타입과 자유 형식 텍스트에 대해 학습된 임베딩으로 합산된 위치 인코딩으로 점과 박스를 나타낸다. 마스크는 convolution을 사용하여 임베딩되고 이미지 임베딩과 함께 element-wise하게 합산된다. 

#### Mask decoder
마스크 디코더는 이미지 임베딩, 프롬프트 임베딩, 출력 토큰을 마스크에 효율적으로 매핑한다. Transformer 디코더 블록을 수정하고 dynamic mask prediction head를 사용한다. 수정된 디코더 블록은 모든 임베딩을 업데이트하기 위해 Prompt Self-Attention과 Cross-Attention을 두 방향(Prompt-to-Image Embedding과 그 반대)으로 사용한다. 두 블록을 실행한 후 이미지 임베딩을 업샘플링하고 MLP는 출력 토큰을 dynamic linear classifier로 매핑한 다음 각 이미지 위치에서 마스크 전경 확률을 계산한다.

#### Resolving ambiguity
모델은 모호한 프롬프트가 주어지면 여러 개의 유효한 마스크를 하나의 출력으로 평균화한다. 이를 해결하기 위해 단일 프롬프트에 대해 여러 출력 마스크를 예측하도록 모델을 수정한다. 저자들은 3개의 마스크 출력이 대부분의 일반적인 경우를 처리하기에 충분하다는 것을 발견했다. 학습 중에는 마스크 중 최소 loss만 backprop한다. 마스크 순위를 매기기 위해 모델은 각 마스크에 대한 신뢰도 점수(ex. 추정된 IoU)를 예측한다.

#### Efficiency
전반적인 모델 디자인은 주로 효율성에 의해 동기가 부여된다. 미리 계산된 이미지 임베딩이 주어지면 프롬프트 인코더와 마스크 디코더가 50ms 내에 CPU의 웹 브라우저에서 실행된다. 이 런타임 성능은 모델의 원활한 실시간으로 상호 작용하는 프롬프트를 가능하게 한다. 

#### Losses and training
[DETR](https://arxiv.org/abs/2005.12872)에서 사용된 focal loss와 dice loss의 선형 결합으로 마스크 예측을 supervise한다. 기하학적 프롬프트의 혼합을 사용하여 promptable segmentation task를 위해 학습된다. SAM이 데이터 엔진에 원활하게 통합될 수 있도록 마스크당 11라운드에서 임의로 프롬프트를 샘플링하여 interactive 설정을 시뮬레이션한다. 

## Segment Anything Data Engine
Segmentation mask가 인터넷에 풍부하지 않기 때문에 저자들은 11억 개의 마스크 데이터셋인 SA-1B를 수집할 수 있는 데이터 엔진을 구축했다. 데이터 엔진은 세 단계로 구성된다.

1. Model-assisted 주석을 사용하는 수동 단계
2. 자동으로 예측된 마스크와 model-assisted 주석이 혼합된 반자동 단계
3. 완전 자동 단계

모델은 주석 입력 없이 마스크를 생성한다. 

#### Assisted-manual stage
첫 번째 단계에서는 고전적인 interactive segmentation과 유사하며 전문 주석 팀이 SAM에서 제공하는 브라우저 기반 interactive segmentation 도구를 사용하여 전경/배경 개체 지점을 클릭하여 마스크에 레이블을 지정했다. 픽셀 정밀 "브러시"와 "지우개" 도구를 사용하여 마스크를 다듬을 수 있다. Model-assisted 주석은 사전 계산된 이미지 임베딩을 사용하여 브라우저 내에서 직접 실시간으로 실행되어 진정한 상호 작용하는 경험을 가능하게 한다. 저자들은 개체에 레이블을 지정하는 데 의미론적 제약을 부과하지 않았으며 주석자는 "물건"과 "사물" 모두에 자유롭게 레이블을 지정했다. 주석 작성자는 눈에 띄는 순서대로 개체에 레이블을 지정하라는 요청을 받았다고 하며 마스크가 주석을 추가하는 데 30초 이상이 걸리면 다음 이미지로 진행하도록 권장되었다.

이 단계의 시작에서 SAM은 공개된 segmentation 데이터셋을 사용하여 학습을 받았다. 충분한 데이터 주석 후 SAM은 새로 주석이 달린 마스크만 사용하여 재학습되었다. 더 많은 마스크가 수집됨에 따라 이미지 인코더가 ViT-B에서 ViT-H로 확장되었으며 기타 아키텍처 세부 사항이 발전했다. 총 6번 모델을 재학습했다. 모델이 개선됨에 따라 마스크당 평균 주석 시간이 34초에서 14초로 감소했으며, 이미지당 평균 마스크 수가 20개에서 44개 마스크로 증가했다. 전반적으로 이 단계에서 12만 개의 이미지에서 430만 개의 마스크를 수집했다. 

#### Semi-automatic stage
이 단계에서 저자들은 모델이 무엇이든 분할하는 능력을 향상시키기 위해 마스크의 다양성을 높이는 것을 목표로 했다. 눈에 잘 띄지 않는 물체에 주석 작성자의 초점을 맞추기 위해 먼저 신뢰할 수 있는 마스크를 자동으로 감지한다. 그런 다음 이러한 마스크로 미리 채워진 이미지를 주석 작성자에게 제공하고 추가로 주석이 지정되지 않은 개체에 주석을 달도록 요청했다. 신뢰할 수 있는 마스크를 감지하기 위해 일반적인 "객체" 범주를 사용하여 모든 첫 번째 단계 마스크에서 boundary box detector를 학습했다. 이 단계에서 18만 개의 이미지에서 590만 개의 마스크를 추가로 수집했으며 (총 1020만 개의 마스크), 새로 수집된 데이터에 대해 주기적으로 모델을 재학습했다 (5회). 마스크당 평균 주석 시간이 최대 34초로 되돌아갔으며 (자동 마스크 제외), 이러한 개체는 레이블을 지정하기가 더 어려웠기 때문이다. 이미지당 평균 마스크 수는 자동 마스크를 포함하여 44개에서 72개로 증가했다.

#### Fully automatic stage
마지막 단계에서 주석은 완전히 자동으로 이루어진다. 이는 모델의 두 가지 주요 개선 사항으로 인해 가능했다. 

1. 첫째, 이 단계를 시작할 때 이전 단계의 다양한 마스크를 포함하여 모델을 크게 개선할 수 있는 충분한 마스크를 수집했다. 
2. 이 단계에서 모호한 경우에도 유효한 마스크를 예측할 수 있는 모호성 인식 모델을 개발했다. 

구체적으로, 32$\times$32 regular grid의 점으로 모델을 유도했고 각 점에 대해 유효한 객체에 해당할 수 있는 마스크 세트를 예측했다. 모호성 인식 모델을 사용하면 점이 부분 또는 하위 부분에 있으면 모델이 하위 부분, 부분 및 전체 개체를 반환한다. 모델의 IoU 예측 모듈은 신뢰할 수 있는 마스크를 선택하는 데 사용된다. 또한 안정적인 마스크만 식별하고 선택한다. 마지막으로 자신 있고 안정적인 마스크를 선택한 후 Non-Maximal Suppression (NMS)를 적용하여 중복을 필터링한다. 더 작은 마스크의 품질을 더욱 향상시키기 위해 여러 개의 겹치는 확대 이미지 crop도 처리하였다. 데이터셋의 모든 1,100만 개 이미지에 완전 자동 마스크 생성을 적용하여 총 11억 개의 고품질 마스크를 생성했다. 

## Segment Anything Dataset
<center><img src='{{"/assets/img/segment-anything/segment-anything-fig2.PNG" | relative_url}}' width="90%"></center>
<br>
SA-1B 데이터셋은 데이터 엔진으로 수집된 1,100만 개의 다양한 고해상도 라이선스 및 개인 정보 보호 이미지와 11억 개의 고품질 segmentation mask로 구성된다. 마스크의 99.1%는 자동으로 생성되었다고 한다. 

저자들은 마스크 품질을 측정하기 위해 랜덤하게 선택된 500개의 이미지와 약 5만 개의 마스크를 주석 전문가들에게 개선해달라고 요청하였다. 주석 전문가들은 편집 도구를 이용하여 정교하게 품질을 개선하였다. 그런 다음 원래의 마스크와 개선된 마스크 사이의 IoU를 계산하였다. 그 결과 94%의 마스크가 90% 이상의 IoU를 보였으며, 97%의 마스크가 75% 이상의 IoU를 보였다고 한다. 

#### Mask properties
다음은 SA-1B의 물체 중심의 공간적 분포를 다른 segmentation 데이터셋과 비교한 것이다.

<center><img src='{{"/assets/img/segment-anything/segment-anything-fig5.PNG" | relative_url}}' width="70%"></center>
<br>
SA-1B는 가장 유사하게 분산된 두 개의 데이터셋인 LVIS v1과 ADE20K에 비해 이미지 모서리의 범위가 더 넓은 반면 COCO와 Open Images V5는 중심 편향이 더 두드러진다. 

다음은 데이터셋들을 크기로 비교한 그래프이다. 

<center><img src='{{"/assets/img/segment-anything/segment-anything-fig6.PNG" | relative_url}}' width="100%"></center>
<br>
예측한대로 SA-1B는 이미지 당 마스크의 개수가 높으며, 중소형 크기의 마스크를 더 많이 포함하는 경향이 있다. 마스크의 오목함 분포가 다른 데이터셋의 오목함 분포와 대체로 유사하다.

## Segment Anything RAI Analysis
다음은 SA-1B 이미지들의 지리적 분포를 나타낸 것이다. 

<center><img src='{{"/assets/img/segment-anything/segment-anything-fig7.PNG" | relative_url}}' width="100%"></center>
<br>
다음은 SA-1B의 지리적 및 소득 표현을 비교한 표이다.

<center><img src='{{"/assets/img/segment-anything/segment-anything-table1.PNG" | relative_url}}' width="55%"></center>
<br>
다음은 성별, 나이, 피부색에 따른 SA-1B의 segmentation 성능을 나타낸 표이다.

<center><img src='{{"/assets/img/segment-anything/segment-anything-table2.PNG" | relative_url}}' width="52%"></center>

## Zero-Shot Transfer Experiments
다음은 SAM의 zero-shot trasfer 능력을 평가하기 위해 사용된 23개의 다양한 데이터셋의 샘플들이다.

<center><img src='{{"/assets/img/segment-anything/segment-anything-fig8.PNG" | relative_url}}' width="100%"></center>

### 1. Zero-Shot Single Point Valid Mask Evaluation
다음은 23개의 데이터셋에 대하여 mIoU를 측정하고 [RITM](https://arxiv.org/abs/2102.06583)과 그 결과를 비교한 그래프들이다. 

<center><img src='{{"/assets/img/segment-anything/segment-anything-fig9.PNG" | relative_url}}' width="100%"></center>

### 2. Zero-Shot Edge Detection
다음은 BSDS500에서 zero-shot으로 edge를 예측한 예시이다.

<center><img src='{{"/assets/img/segment-anything/segment-anything-fig10.PNG" | relative_url}}' width="65%"></center>
<br>
다음은 BSDS500에서 edge detection에 대한 zero-shot transfer 성능을 비교한 표이다. 

<center><img src='{{"/assets/img/segment-anything/segment-anything-table3.PNG" | relative_url}}' width="55%"></center>

### 3. Zero-Shot Object Proposals
다음은 LVIS v1에서 object proposal 생성 성능을 비교한 표이다.

<center><img src='{{"/assets/img/segment-anything/segment-anything-table4.PNG" | relative_url}}' width="57%"></center>

### 4. Zero-Shot Instance Segmentation
다음은 instance segmentation 결과를 나타낸 표이다.

<center><img src='{{"/assets/img/segment-anything/segment-anything-table5.PNG" | relative_url}}' width="55%"></center>
<br>
다음은 사람이 평가한 마스크 품질 rating을 나타낸 그래프이다.

<center><img src='{{"/assets/img/segment-anything/segment-anything-fig11.PNG" | relative_url}}' width="60%"></center>

### 5. Zero-Shot Text-to-Mask
다음은 zero-Shot text-to-mask에 대한 정량적 결과이다. 

<center><img src='{{"/assets/img/segment-anything/segment-anything-fig12.PNG" | relative_url}}' width="60%"></center>
<br>
SAM은 간단하고 미묘한 텍스트 프롬프트에 대하여 동작할 수 있으며, 정확한 예측에 실패한 경우 추가 점 프롬프트로 예측을 도울 수 있다. 

### 6. Ablations
다음은 데이터 엔진 단계(왼쪽), 학습 데이터 스케일링(중간), 이미지 인코더 스케일링(오른쪽)에 대한 ablation study 결과를 나타낸 그래프이다. 

<center><img src='{{"/assets/img/segment-anything/segment-anything-fig13.PNG" | relative_url}}' width="100%"></center>