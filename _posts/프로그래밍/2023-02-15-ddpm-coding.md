---
title: "[구현] Colab에서 DDPM 구현하기 - 진행상황"
last_modified_at: 2023-02-15
categories:
  - 프로그래밍
tags:
  - Diffusion
  - Computer Vision
  - AI
excerpt: "Colab에서 DDPM 구현하기 - 진행상황"
use_math: true
classes: wide
---

> 다음 글: [다양한 결과 모음](https://kimjy99.github.io/프로그래밍/ddpm-coding-2/)

### 1월 22일
첫 학습을 4일만에 종료하였다. 총 19시간 정도 학습하였다. 

> 이미지 전처리: CelebA 178 $\times$ 218 → CenterCrop 178 $\times$ 178 → Resize 156 $\times$ 156 → RandomCrop 128 $\times$ 128  
> U-Net 채널 수: 64, 64, 128, 128
> Batch size: 120 (mini batch size 12 $\times$ accumulation step 10)  
> Learning rate: $6 \times 10^{-5}$  
> beta schedule: linear  
> 총 iteration: 56,000 (Gradient update는 5600번)  

생각보다 GPU 메모리를 많이 잡아먹는다. Mini batch size를 16으로만 해도 CUDA out of memory 에러가 떠서 12로 줄였다. 

DDPM 논문에서 256$\times$256 CelebA-HQ 모델의 경우 batch size 64, learning rate $2 \times 10^{-5}$으로 설정하고 있다. Batch size를 2배 정도 크고 이미지 크기가 절반이므로 learning rate를 3배 정도 올려주었다.

Loss는 0.02 정도까지 쭉 감소한 후 더 이상 줄어들지 않아 학습을 끝냈다. 

$T = 1000$으로 12장을 한 번 샘플링하는 데 6분 정도 걸린다. 샘플은 아래와 같다. 

<center><img src='{{"/assets/img/ddpm-coding/ddpm-coding-fig1.png" | relative_url}}' width="50%"></center>
<br>
특정 색이 너무 강조되어 나오고 있고, 얼굴 모양 자체도 이상하다. 아무래도 학습이 덜 진행된 것 같은 데 더 이상 loss가 떨어지지 않는다. 

### 2월 3일

> 이미지 전처리: CelebA 178 $\times$ 218 → CenterCrop 178 $\times$ 178 → Resize 144 $\times$ 144 → RandomCrop 128 $\times$ 128  
> U-Net 채널 수: 64, 64, 128, 128  
> Batch size: 16 (mini batch size 8 $\times$ accumulation step 2)  
> Learning rate: $1 \times 10^{-4}$  
> beta schedule: linear  
> P2-weighting 사용  
> 총 iteration: 100,000 (Gradient update는 50,000번)  

크게 4가지를 변경하여 다시 학습을 진행하였다. 

1. **P2-weighting** ([논문리뷰]() 참고)을 사용하였다. 단순히 loss항에 $1-\bar{\alpha}$를 가중치로 주면 되기 때문에 큰 수정이 필요 없다. 
2. 이미지 전처리 시 156 $\times$ 156으로 resize를 하면 128 $\times$ 128로 랜덤하게 crop되는 과정에서 얼굴이 아닌 부분이 많이 포함되는 경우가 존재했다. 따라서 156 $\times$ 156 대신 144 $\times$ 144로 resize하였다. 
3. 기존의 batch size가 지나치게 커서 gradient update가 너무 적게 되고 학습이 너무 오래 걸리고 있었다. 따라서 batch size를 16으로 줄였다. 
4. 이에 맞춰 learning rate를 $1 \times 10^{-4}$로 키웠다. 

7일이 소요되었다. 총 33시간 정도 학습을 진행하였다. 

Loss가 0.0023 정도까지 감소하였다. P2-weighting 가중치의 평균이 0.7368인 것을 고려해도 많이 감소하였다. 

샘플은 아래와 같다. 

<center><img src='{{"/assets/img/ddpm-coding/ddpm-coding-fig2.png" | relative_url}}' width="50%"></center>
<br>
U-Net 채널 수를 고려하면 나름 괜찮은 결과가 나온 것 같다. 

### 2월 12일
2월 6일 모델에 이어서 학습을 진행하였다. Learning rate를 5만 iteration마다 절반으로 줄였다. 다른 세팅은 그대로 두었다.

> Learning rate: 10만 번째 iteration부터 $5 \times 10^{-5}$, 15만 번째 iteration부터 $2.5 \times 10^{-5}$  
> 총 iteration: 200,000 (Gradient update는 10만 번)  

추가로 7일이 더 소요되었고, 총 66시간 정도 학습이 진행되었다. 

Learning rate가 절반이 되면 10,000 iteration 정도 loss가 소폭 감소하다가 감소를 멈추었으며 최종적으로 0.0021까지 감소하였다. 

샘플은 아래와 같다. 

<center><img src='{{"/assets/img/ddpm-coding/ddpm-coding-fig3.png" | relative_url}}' width="50%"></center>
<br>
확실히 성능이 개선되었다. 이상한 샘플이 나오는 비율도 줄어든 것 같다. 어떤 논문에서 loss가 거의 줄어들지 않더라도 오래 학습하는 것이 성능 개선에 효과가 있다고 하던데 오래 학습하는 것이 효과가 있긴 한 것 같다.

약 3시간 동안 360개의 샘플을 생성하고 그 중 괜찮은 샘플만 모아보았다. 

<center><img src='{{"/assets/img/ddpm-coding/samples.png" | relative_url}}' width="100%"></center>
<br>
모아놓고 보니 꽤 그럴듯하다. 아무래도 CelebA 데이터셋이 여성 얼굴이 1.4배 더 많기 때문에 여성 얼굴을 더 많이 생성한다. 

### 앞으로

AFHQ 데이터셋에 대하여 학습을 진행해볼까 한다. 아무래도 사람 얼굴보다는 개나 고양이 얼굴이 덜 다양하기 때문에 작은 아키텍처에서 좋은 결과가 나오지 않을까 싶다. 아니면 CelebA의 배경을 제거한 데이터셋을 구축하여 학습을 진행하면 모델이 좀 더 얼굴 생성에 집중하지 않을까 싶기도 하다. 