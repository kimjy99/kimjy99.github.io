---
title: "[논문리뷰] SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer"
last_modified_at: 2026-08-14
categories:
  - 논문리뷰
tags:
  - Computer Vision
  - Image Generation
  - Image Tokenization
  - CVPR
excerpt: "SoftVQ-VAE 논문 리뷰 (CVPR 2025)"
use_math: true
classes: wide
---

> CVPR 2025. [[Paper](https://arxiv.org/abs/2412.10958)] [[Github](https://github.com/deepseek-ai/DeepSpec)]  
> Hao Chen, Ze Wang, Xiang Li, Ximeng Sun, Fangyi Chen, Jiang Liu, Jindong Wang, Bhiksha Raj, Zicheng Liu, Emad Barsoum  
> Carnegie Mellon University | AMD | William & Mary | MBZUAI  
> 14 Dec 2024  

<center><img src='{{"/assets/img/softvq-vae/softvq-vae-fig1.webp" | relative_url}}' width="100%"></center>

## Introduction
KL-VAE와 VQ-VAE는 많은 생성 모델에서 주로 채택되었지만, 여전히 생성 모델링의 효율성과 효과를 제한하는 두 가지 주요 문제점을 안고 있다.

1. 더 높은 압축률을 달성하기 어렵다.
2. 다른 self-supervised 방식에 비해 latent space에서 판별력이 떨어진다.

특히 Transformer 기반 아키텍처의 효율성은 latent 토큰 수에 대한 제곱 복잡도에 의해 근본적으로 제한된다. 현재의 이미지 tokenizer는 일반적으로 256$\times$256 이미지를 최소 256개의 토큰으로, 512$\times$512 이미지를 최소 1024개의 토큰으로 압축하여 생성 모델의 학습과 inference 모두에 상당한 계산 병목 현상을 초래한다. 최근 tokenizer의 토큰 수를 근본적으로 줄이는 연구가 활발히 진행되고 있다. 그러나 압축률을 크게 높이면 재구성 품질이 크게 저하되어 생성 성능이 저하된다.

압축 문제 외에도, 현재의 tokenizer는 일반적으로 판별력이 높은 feature를 포착하는 데 어려움을 겪는다. 최근에는 latent 토큰의 표현 정렬에 대한 연구도 진행되었다. 그러나 KL-VAE의 smoothness 제약 조건과 VQVAE의 discrete quantization으로 인한 손실 압축 특성 때문에, 현재의 tokenizer는 latent space의 semantic 정보를 학습하는 데 한계를 보인다.

본 논문에서는 VQ-VAE를 간단하게 변형한 **SoftVQ-VAE**를 제안하였다. SoftVQ-VAE는 기존의 discrete tokenizer를 높은 압축률과 풍부한 semantic의 latent space를 갖는 continuous tokenizer로 변환하였다. 구체적으로, 학습 가능한 codebook을 사용하는 soft categorical posterior를 VAE에 적용하는 것을 제안하였다. 기존 VQ-VAE에서 토큰과 codeword 간의 일대일 매핑 대신, SoftVQ-VAE는 각 latent 토큰에 여러 codeword를 적응적으로 집계할 수 있도록 하여 latent space의 표현 용량을 크게 향상시킨다.

본 논문에서는 SoftVQ-VAE를 Transformer 오토인코더 아키텍처에 적용하여 재구성 및 생성 모두에서 훨씬 적은 수의 1D latent 토큰(32, 64)을 사용하는 데 성공했다. SoftVQ-VAE는 완전 미분 가능하므로, VQ-VAE에서처럼 codebook loss나 commit loss가 필요하지 않아 codebook 학습이 간소화될 뿐만 아니라, 간단한 코사인 유사도 loss를 사용하여 사전 학습된 semantic feature와 직접 정렬함으로써 더 나은 표현 학습을 가능하게 한다.

## Method
### 1. Architecture
본 논문에서는 SoftVQ-VAE의 인코더 $\mathcal{E}$와 디코더 $\mathcal{D}$에 ViT 아키텍처를 활용하였다. [TiTok](https://kimjy99.github.io/논문리뷰/titok)과 유사하게, 이미지 feature를 latent로 사용하는 대신, 추가적인 학습 가능한 1D 토큰 세트를 초기화하고 이러한 토큰을 재구성 및 생성에 사용한다. Self-attention 메커니즘을 통해 학습 가능한 토큰은 다양한 이미지 토큰을 적응적으로 통합하여 SoftVQ의 latent 토큰을 생성할 수 있다.

구체적으로, 패치 크기가 $P$인 인코더의 경우, 이미지 $\textbf{x} \in \mathbb{R}^{H \times W \times 3}$는 이미지 토큰 시퀀스 $$\textbf{x}_p \in \mathbb{R}^{N \times D}$$로 patchify된다 ($N = HW/P^2$). 그런 다음 학습 가능한 토큰 집합 $$\textbf{z}_l \in \mathbb{R}^{L \times D}$$을 이미지 토큰과 함께 인코더의 입력으로 concat하고 학습 가능한 latent 토큰에 해당하는 출력만 인코더의 출력 $$\hat{\textbf{z}} = \mathcal{E}([\textbf{x}_p; \textbf{z}_l]; \phi)$$으로 유지한다.

디코더의 경우, 학습 가능한 마스크 토큰 시퀀스 $$\textbf{m} \in \mathbb{R}^{N \times D}$$와 latent 토큰 $\textbf{z}$를 concat하여 이미지 $$\hat{\textbf{x}} = \mathcal{D}([\textbf{m}; \textbf{z}]; \theta)$$를 재구성한다. 디코더의 마지막에는 linear layer를 사용하여 마스크 토큰으로부터 픽셀 값을 예측한다. 인코더 입력의 이미지 토큰과 디코더 입력의 마스크 토큰에는 각각 2D absolute position embedding을 적용하고, latent 토큰에는 1D absolute position embedding을 적용한다.

이러한 디자인 덕분에 SoftVQ-VAE는 이미지 토큰과 적응적으로 연결된 임의의 길이의 latent 코드를 재구성 및 생성 모델링에 사용할 수 있으며, 동일한 길이의 latent 코드를 사용하여 다양한 해상도의 이미지를 모델링할 수 있다.

<center><img src='{{"/assets/img/softvq-vae/softvq-vae-fig2.webp" | relative_url}}' width="60%"></center>

### 2. SoftVQ-VAE
KL-VAE와 VQ-VAE 모두 높은 압축률로 인해 재구성 및 latent space의 품질이 크게 저하되는 경우가 많다. 이러한 한계를 극복하기 위해, 본 논문에서는 VQ-VAE를 간단하게 변형한 **SoftVQ-VAE**를 제안하였다. SoftVQ-VAE는 데이터 분포를 포착하기 위해 codeword를 학습하는 장점을 유지하면서 continuous tokenizer로서 더 높은 표현 용량을 제공한다. SoftVQ-VAE의 핵심 아이디어는 각 latent code가 학습 가능한 codebook에서 여러 codeword를 적응적으로 통합할 수 있도록 하는 것이다. 결과적으로, 이는 인코더 posterior에 soft categorical distribution을 적용한다.

$$
\begin{aligned}
\textrm{posterior}: & \; q_\phi (\textbf{z} \vert \textbf{x}) = \textrm{Softmax} \left( - \frac{\| \hat{\textbf{z}} - \mathcal{C} \|_2 }{\tau} \right) \\
\textrm{latent}: & \; \textbf{z} = q_\phi (\textbf{z} \vert \textbf{x}) \mathcal{C} \\
\textrm{kl}: & \; \mathcal{L}_\textrm{kl} = H(q_\phi (\textbf{z} \vert \textbf{x})) - H(\mathbb{E}_{\textbf{x} \sim p(\textbf{x})} q_\phi (\textbf{z} \vert \textbf{x})) \\
\end{aligned}
$$

(Temperature $\tau = 0.07$)

이러한 간단한 수정만으로도 SoftVQ-VAE는 재구성 품질과 latent space의 높은 품질을 유지하면서 latent 토큰 길이를 크게 줄일 수 있다. SoftVQ-VAE는 완전 미분 가능하므로 인코더와 codebook을 직접 최적화할 수 있으며, latent space에 다양한 형태의 정규화를 더 쉽게 적용할 수 있고, codebook loss와 commit loss가 더 이상 필요 없다.

### 3. Representation Alignment of Latent Space
Tokenizer의 재구성 품질도 중요하지만, 후속적인 denoising 기반 생성 모델링을 위해서는 고품질의 latent space를 학습시키는 것이 더욱 중요하다. 그러나 VQ-VAE의 discrete quantization과 KL-VAE의 Gaussian 제약 조건으로 인해 latent space에 효과적인 정규화를 적용하는 것은 여전히 ​​어렵다.

SoftVQ-VAE의 완전 미분 가능성 덕분에 이제 latent space에 직접 정규화를 적용할 수 있다. 저자들은 [REPA](https://kimjy99.github.io/논문리뷰/repa)에서 영감을 받아, latent 코드의 표현을 사전 학습된 [DINOv2](https://kimjy99.github.io/논문리뷰/dinov2) 비전 인코더와 정렬하는 방법을 제안하였다. 생성 모델의 중간 layer에서 표현을 정렬하는 REPA와 달리, 본 논문에서는 tokenizer의 latent space에서 feature를 정렬하며, 생성 모델의 입력 공간에서 REPA를 수행하는 것과 동일하다. 구체적으로, latent 코드를 사전 학습된 비전 인코더의 이미지 토큰과 정렬하기 위해 각 latent 토큰을 $N/L$번 복제한다.

$$
\begin{equation}
\textbf{z}_r = [\underbrace{\textbf{z}^{[0]}, \ldots, \textbf{z}^{[0]}}_{N/L \textrm{ times}}, \underbrace{\textbf{z}^{[1]}, \ldots, \textbf{z}^{[1]}}_{N/L \textrm{ times}}, \ldots, \underbrace{\textbf{z}^{[L]}, \ldots, \textbf{z}^{[L]}}_{N/L \textrm{ times}}]
\end{equation}
$$

사전 학습된 feature $$\textbf{y}_\ast$$와 매칭시키기 위해 latent 토큰 위에 MLP를 적용한다.

$$
\begin{equation}
\mathcal{L}_\textrm{align} = \frac{1}{N} \sum_{n=1}^N \textrm{sim} \left( \textbf{y}_\ast^{[n]}, \textrm{MLP} \left( \textbf{z}_r^{[n]} \right) \right)
\end{equation}
$$

이러한 정렬은 SoftVQ-VAE의 latent space가 semantic하게 구별되는 feature를 포착하도록 보장하며, 이는 tokenizer의 재구성 성능 향상으로 직접 이어지지 않더라도 생성 모델 학습에 도움이 된다. 학습된 생성 모델의 생성 품질은 tokenizer의 재구성 능력보다는 latent space의 semantic 구조에 더 크게 의존하는 경우가 많다.

### 4. Final Training Objective
SoftVQ-VAE의 학습 loss는 reconstruction loss, perceptual loss, adversarial loss, 표현 정렬 loss를 결합한 것이다.

$$
\begin{equation}
\mathcal{L} = \mathcal{L}_\textrm{recon} + \lambda_1 \mathcal{L}_\textrm{percep} + \lambda_2 \mathcal{L}_\textrm{adv} + \lambda_3 \mathcal{L}_\textrm{align} + \lambda_4 \mathcal{L}_\textrm{KL}
\end{equation}
$$

($$\lambda_1 = 1.0$$, $$\lambda_2 = 0.2$$, $$\lambda_3 = 0.1$$, $$\lambda_4 = 0.01$$)

## Experiments
- Configuration
  - SoftVQ-VAE: SoftVQ-S (45M), SoftVQ-B (173M), SoftVQ-BL (391M), SoftVQ-L (608M)
   - 각 configuration마다 $L = 32$와 $L = 64$ variant
  - 생성 모델: [DiT](https://kimjy99.github.io/논문리뷰/dit), [SiT](https://arxiv.org/abs/2408.12245), [MAR](https://kimjy99.github.io/논문리뷰/mar)

### 1. Main Results
다음은 ImageNet 256$\times$256 조건부 생성에 대한 시스템 수준의 비교 결과이다.

<center><img src='{{"/assets/img/softvq-vae/softvq-vae-table1.webp" | relative_url}}' width="88%"></center>
<br>
다음은 ImageNet 512$\times$512 조건부 생성에 대한 시스템 수준의 비교 결과이다.

<center><img src='{{"/assets/img/softvq-vae/softvq-vae-table2.webp" | relative_url}}' width="88%"></center>

### 2. Comparison of Tokenizers
다음은 ImageNet 256$\times$256 클래스 조건부 생성에서 tokenizer를 비교한 결과이다.

<center><img src='{{"/assets/img/softvq-vae/softvq-vae-table3.webp" | relative_url}}' width="57%"></center>

### 3. Discussions on the Latent Space
다음은 표현 정렬에 대한 비교 결과이다. (SoftVQ-B 64 토큰)

<center><img src='{{"/assets/img/softvq-vae/softvq-vae-table4.webp" | relative_url}}' width="52%"></center>
<br>
다음은 ImageNet-1K validation set에 대한 linear probing 정확도를 비교한 결과이다.

<center><img src='{{"/assets/img/softvq-vae/softvq-vae-fig3.webp" | relative_url}}' width="60%"></center>
<br>
다음은 인코더 출력 $$\hat{\textbf{z}}$$ 디코더 입력 $\textbf{z}$를 시각화한 것이다.

<center><img src='{{"/assets/img/softvq-vae/softvq-vae-fig4.webp" | relative_url}}' width="90%"></center>

### 4. Ablation Studies
다음은 ablation study 결과이다.

<center><img src='{{"/assets/img/softvq-vae/softvq-vae-table6.webp" | relative_url}}' width="66%"></center>