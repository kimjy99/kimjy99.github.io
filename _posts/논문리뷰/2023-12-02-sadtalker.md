---
title: "[논문리뷰] SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation"
last_modified_at: 2023-12-02
categories:
  - 논문리뷰
tags:
  - Talking Head
  - Video Generation
  - Computer Vision
  - AI
  - CVPR
excerpt: "SadTalker 논문 리뷰 (CVPR 2023)"
use_math: true
classes: wide
---

> CVPR 2023. [[Paper](https://arxiv.org/abs/2211.12194)] [[Page](https://sadtalker.github.io/)] [[Github](https://github.com/OpenTalker/SadTalker)]  
> Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, Fei Wang  
> Xi’an Jiaotong University | Tencent AI Lab | Ant Group  
> 22 Nov 2022  

<center><img src='{{"/assets/img/sadtalker/sadtalker-fig1.PNG" | relative_url}}' width="90%"></center>

## Introduction
음성 오디오를 사용하여 정적 인물 이미지를 애니메이션화하는 것은 어려운 작업이며 중요한 애플리케이션이 많다. 이전 연구들은 주로 음성과 강한 연관성이 있는 입술 동작 생성에 중점을 두었다. 최근 연구에서는 머리 포즈와 같은 다른 관련 모션이 포함된 사실적인 말하는 머리 동영상을 생성하는 것을 목표로 하고 있다. 그들의 방법은 주로 랜드마크와 latent warping을 통해 2D 모션 필드를 도입한다. 그러나 생성된 영상의 품질은 여전히 부자연스럽고 선호 포즈, 블러, identity 변경, 왜곡된 얼굴로 인해 제한된다.

자연스럽게 보이는 말하는 머리 동영상을 생성하는 것은 오디오와 다양한 동작 간의 연결이 다르기 때문에 많은 어려움을 안겨준다. 즉, 입술 움직임은 오디오와 가장 강한 연관성을 가지고 있지만 오디오는 다양한 머리 자세와 눈 깜박임을 통해 말할 수 있다. 따라서 이전의 얼굴 랜드마크 기반 방법과 2D flow 기반 오디오-to-표현 네트워크는 머리 동작과 표정이 표현에서 완전히 분리되지 않기 때문에 왜곡된 얼굴을 생성할 수 있다. 또 다른 인기 있는 방법은 latent 기반 얼굴 애니메이션이며, 주로 말하는 머리 애니메이션의 특정 종류의 동작에 초점을 맞추고 고품질 동영상을 합성하기 위해 노력한다. 3D 얼굴 모델은 고도로 분리된 표현을 포함하고 있으며 각 동작 유형을 개별적으로 학습하는 데 사용될 수 있다. 하지만 해당 방법은 부정확한 표현과 부자연스러운 모션 시퀀스도 생성한다.

위의 관찰로부터 본 논문은 암시적 3D 계수 변조를 통해 **S**tylized **A**udio-**D**riven **Talk**ing-head 동영상 생성 시스템인 **SadTalker**를 제안한다. 이를 위해 3DMM의 모션 계수를 중간 표현으로 간주하고 task를 두 가지 주요 구성 요소로 나눈다. 한편으로 오디오에서 현실적인 모션 계수 (ex. 머리 포즈, 입술 모션, 눈 깜박임)를 생성하고 각 모션을 개별적으로 학습하여 불확실성을 줄이는 것을 목표로 한다. 표현을 위해 재구성된 렌더링된 3D 얼굴의 입술 모션 계수와 perceptual loss들 (lip reading loss, 얼굴 랜드마크 loss)의 계수를 증류하여 새로운 오디오-to-표현 계수 네트워크를 설계한다. Stylize된 머리 포즈의 경우 조건부 VAE를 사용하여 주어진 포즈의 잔차를 학습하여 다양성과 실제와 같은 머리 모션을 모델링한다. 사실적인 3DMM 계수를 생성한 후 새로운 3D-aware 얼굴 렌더링을 통해 소스 이미지를 가이드한다. Face-vid2vid에서 영감을 받아 명시적 3DMM 계수와 unsupervised 3D 키포인트 도메인 간의 매핑을 학습한다. 그런 다음 소스 이미지와 가이드 이미지의 unsupervised 3D 키포인트를 통해 워핑 필드를 생성하고 레퍼런스 이미지를 워핑하여 최종 동영상을 생성한다. 표정 생성, 머리 포즈 생성, 얼굴 렌더러의 각 하위 네트워크를 개별적으로 학습하고 end-to-end 스타일로 inference할 수 있다.

## Method
<center><img src='{{"/assets/img/sadtalker/sadtalker-fig2.PNG" | relative_url}}' width="75%"></center>
<br>
위 그림에서 볼 수 있듯이 본 논문의 시스템은 말하는 머리 생성을 위한 중간 표현으로 3D 모션 계수를 사용한다. 먼저 원본 이미지에서 계수를 추출한다. 그런 다음 ExpNet과 PoseVAE에 의해 현실적인 3DMM 모션 계수가 개별적으로 생성된다. 마지막으로 말하는 머리 동영상을 제작하기 위해 3D-aware 얼굴 렌더링을 사용한다.

### 1. Preliminary of 3D Face Model
실제 동영상은 3D 환경에서 캡처되기 때문에 생성된 동영상의 현실감을 높이기 위해서는 3D 정보가 매우 중요하다. 그러나 이전 연구들은 단일 이미지에서 정확한 3차원 계수를 얻기 어렵고, 고품질의 얼굴 렌더 디자인도 어렵기 때문에 3차원 공간에서는 거의 고려되지 않았다. 최근 제안된 [단일 이미지 3D 재구성 방법](https://arxiv.org/abs/1903.08527)에서 영감을 받아 예측된 3D Morphable Model (3DMM)의 공간을 중간 표현으로 간주한다. 3DMM에서 3D 얼굴 모양 $S$는 다음과 같이 분리될 수 있다.

$$
\begin{equation}
S = \bar{S} + \alpha U_\textrm{id} + \beta U_\textrm{exp}
\end{equation}
$$

여기서 $\bar{S}$는 3D 얼굴의 평균 모양이고 $$U_\textrm{id}$$와 $$U_\textrm{exp}$$는 LSFM morphable model의 식별 및 표현에 대한 정규 직교 기저 (orthonormal basis)이다. 계수 $\alpha \in \mathbb{R}^{80}$과 $\beta \in \mathbb{R}^{64}$는 각각 identity와 표현을 나타낸다. 포즈 변화를 보존하기 위해 계수 $r \in SO(3)$과 $t \in \mathbb{R}^3$는 머리 회전과 평행이동을 나타낸다. Identity에 관련 없는 계수 생성을 위해 모션 파라미터만 $$\{\beta, r, t\}$$로 모델링한다. 가이드 오디오로부터 머리 포즈 $\rho = [r, t]$와 표현 계수 $\beta$를 개별적으로 학습한다. 그런 다음 이러한 모션 계수는 최종 동영상 합성을 위해 얼굴 렌더링을 암시적으로 변조하는 데 사용된다.

### 2. Motion Coefficients Generation through Audio
3D 모션 계수에는 머리 포즈와 표현이 모두 포함되어 있으며 머리 포즈는 글로벌 모션이고 표현은 상대적으로 로컬하다. 이를 위해 모든 것을 학습하면 머리 포즈는 오디오와의 관계가 상대적으로 약하고 입술 동작은 연결성이 높기 때문에 네트워크에 큰 불확실성이 발생한다. 본 논문은 PoseVAE와 ExpNet을 사용하여 머리 포즈와 표현의 모션을 생성한다.

#### ExpNet
오디오에서 정확한 표현 계수를 생성하는 일반 모델을 학습하는 것은 다음 두 가지 이유로 매우 어렵다. 

1. 오디오-to-표현은 서로 다른 identity에 대한 일대일 매핑 task가 아니다. 
2. 표현 계수에 오디오와 관련 없는 모션이 일부 있으며 이는 예측의 정확도에 영향을 미친다. 

ExpNet은 이러한 불확실성을 줄이기 위해 설계되었다. Identity 문제에 대해서는 첫 번째 프레임의 표정 계수 $\beta_0$을 통해 표정 모션을 특정 인물과 연결한다. 자연스러운 대화에서 다른 얼굴 성분의 모션 가중치를 줄이기 위해 사전 학습된 [Wav2Lip](https://arxiv.org/abs/2008.10010) 네트워크와 [deep 3D reconstruction](https://arxiv.org/abs/1903.08527)을 통해 입술 모션 전용 계수를 계수 타겟으로 사용한다. 그런 다음 렌더링된 이미지의 추가 랜드마크 loss를 통해 다른 사소한 얼굴 모션 (ex. 눈 깜박임)을 활용할 수 있다.

<center><img src='{{"/assets/img/sadtalker/sadtalker-fig3.PNG" | relative_url}}' width="50%"></center>
<br>
위 그림에 표시된 것처럼 오디오 window $$a_{\{1,\ldots,t\}}$$에서 $t$ 프레임의 표현 계수를 생성한다. 여기서 각 프레임의 오디오 feature는 0.2초 mel-spectrogram이다. 학습을 위해 먼저 ResNet 기반 오디오 인코더 $\Phi_A$를 설계하여 latent space에 오디오 feature를 삽입한다. 그런 다음, 표현 계수를 디코딩하기 위해 매핑 네트워크 $\Phi_M$으로 linear layer가 추가된다. 여기서는 위에서 논의한 대로 identity 불확실성을 줄이기 위해 레퍼런스 이미지의 레퍼런스 표현 $\beta_0$도 추가한다. 학습에서 입술 전용 계수를 ground truth로 사용하므로 눈 깜박임 제어 신호 $z_\textrm{blink} \in [0, 1]$와 해당 눈 랜드마크 loss를 명시적으로 추가하여 제어 가능한 눈 깜박임을 생성한다. 네트워크는 다음과 같이 쓸 수 있다.

$$
\begin{equation}
\beta_{\{1,\ldots,t\}} = \Phi_M (\Phi_A (a_{\{1,\ldots,t\}}), z_\textrm{blink}, \beta_0)
\end{equation}
$$

Loss function의 경우 먼저 $$\mathcal{L}_\textrm{distill}$$을 사용하여 입술 전용 표현 계수 $$R_e (\textrm{Wav2Lip} (I_0, a_{\{1,\ldots,t\}}))$$와 생성된 $$\beta_{\{1,\ldots,t\}}$$ 간의 차이를 평가한다. 립싱크 동영상을 생성하기 위해 Wav2Lip의 첫 번째 프레임 $I_0$만 사용하여 포즈 변형과 입술 움직임을 제외한 기타 얼굴 표정의 영향을 줄인다. 게다가, 명시적인 얼굴 모션 공간에서 추가적인 perceptual loss를 계산하기 위해 differentiable 3D face render $R_d$도 포함한다. 랜드마크 loss $$\mathcal{L}_\textrm{lks}$$를 계산하여 눈 깜박임 범위와 전반적인 표정 정확도를 측정한다. 사전 학습된 lip reading 네트워크 $$\Phi_\textrm{render}$$는 입술 품질을 유지하기 위해 시간적 lip reading loss $$\mathcal{L}_\textrm{read}$$로도 사용된다. 

#### PoseVAE
<center><img src='{{"/assets/img/sadtalker/sadtalker-fig4.PNG" | relative_url}}' width="50%"></center>
<br>
위 그림에서 볼 수 있듯이 VAE 기반 모델은 실제 말하는 동영상의 현실적이고 stylize된 머리 움직임 $\rho \in \mathbb{R}^6$을 학습하도록 설계되었다. 학습에서 포즈 VAE는 인코더-디코더 기반 구조를 사용하여 고정된 $n$ 프레임에 대해 학습된다. 인코더와 디코더는 모두 2-layer MLP이며, 입력에는 순차적인 $t$ 프레임의 머리 포즈가 포함되어 있으며 이를 가우시안 분포에 임베딩한다. 디코더에서 네트워크는 샘플링된 분포에서 $t$ 프레임의 포즈를 생성하는 방법을 학습한다. 포즈를 직접 생성하는 대신 PoseVAE는 첫 번째 프레임의 조건 포즈 $\rho_0$의 residual을 학습하므로 첫 번째 프레임 조건에서 테스트할 때 더 길고 안정적이며 지속적인 머리 모션을 생성할 수 있다. 또한 CVAE를 따라 리듬 인식과 identity 스타일의 조건으로 해당 오디오 feature $$a_{\{1,\ldots,t\}}$$와 스타일 identity $Z_\textrm{style}$을 추가한다. KL-divergence $$\mathcal{L}_\textrm{KL}$$은 생성된 모션의 분포를 측정하는 데 사용된다. 평균 제곱 loss $$\mathcal{L}_\textrm{MSE}$$와 adversarial loss $$\mathcal{L}_\textrm{GAN}$$은 생성된 품질을 보장하는 데 사용된다. 

### 3. 3D-aware Face Render
사실적인 3D 모션 계수를 생성한 후 잘 설계된 3D-aware 이미지 애니메이터를 통해 최종 동영상을 렌더링한다. 저자들은 하나의 이미지에서 3D 정보를 암시적으로 학습하는 최신 이미지 애니메이션 방법인 [face-vid2vid](https://arxiv.org/abs/2011.15126)에서 영감을 얻었다. 그러나 이 방법에서는 신호를 가이드하는 모션으로 실제 동영상이 필요하다. 본 논문의 얼굴 렌더링은 3DMM 계수를 통해 가이드 가능하게 만든다. 

<center><img src='{{"/assets/img/sadtalker/sadtalker-fig5.PNG" | relative_url}}' width="60%"></center>
<br>
위 그림에서 볼 수 있듯이 명시적인 3DMM 모션 계수 (머리 포즈, 표현)와 암시적인 unsupervised 3D 키포인트 간의 관계를 학습하기 위해 mappingNet을 제안한다. mappingNet은 여러 1D convolution layer를 통해 구축되었다. 또한 [PIRenderer](https://arxiv.org/abs/2109.08379)와 같이 smoothing을 위해 시간 window의 시간 계수를 사용한다. 저자들은 PIRenderer의 얼굴 정렬 모션 계수는 오디오 기반 동영상 생성의 자연스러운 모션에 큰 영향을 미친다는 것을 확인했다. 본 논문은 표현 계수와 머리 포즈만을 사용한다.

학습의 경우 두 단계가 포함된다. 첫째, 원본 논문에서와 같이 self-supervised 방식으로 Face-vid2vid를 학습한다. 두 번째 단계에서는 튜닝을 위해 appearance encoder, canonical keypoints estimator, image generator의 모든 파라미터를 고정한다. 그런 다음 재구성 스타일로 ground truth 동영상의 3DMM 계수에 대한 매핑 네트워크를 학습한다. 원래 구현에 따라 L1 loss와 최종 생성된 동영상을 사용하여 unsupervised 키포인트 도메인에서 supervision을 제공한다. 

## Experiments
- 데이터셋: VoxCeleb
- 구현 디테일
  - ExpNet, PoseVAE, FaceRender는 각각 학습됨
  - optimizer: 모두 Adam
  - learning rate: $2 \times 10^{-5}$, $1 \times 10^{-4}$, $2 \times 10^{-4}$
  - 학습 프레임 수: 5, 32, 5

### 1. Compare with other state-of-the-art methods
다음은 HDTF 데이터셋에서 SOTA 방법들과 성능을 비교한 표이다. 

<center><img src='{{"/assets/img/sadtalker/sadtalker-table1.PNG" | relative_url}}' width="88%"></center>
<br>
다음은 여러 SOTA 방법들과 생성 결과를 비교한 것이다. 

<center><img src='{{"/assets/img/sadtalker/sadtalker-fig6.PNG" | relative_url}}' width="100%"></center>

### 2. User Studies
다음은 user study 결과이다. 

<center><img src='{{"/assets/img/sadtalker/sadtalker-table2.PNG" | relative_url}}' width="55%"></center>

### 3. Ablation Studies
#### Ablation of ExpNet
다음은 ExpNet에 대한 ablation 결과이다. 

<center><img src='{{"/assets/img/sadtalker/sadtalker-table3.PNG" | relative_url}}' width="50%"></center>
<br>
<center><img src='{{"/assets/img/sadtalker/sadtalker-fig8.PNG" | relative_url}}' width="67%"></center>
<br>
다음은 조건 없이 단일 네트워크에서 모든 계수를 학습하는 baseline과 SadTalker를 비교한 것이다.

<center><img src='{{"/assets/img/sadtalker/sadtalker-fig7.PNG" | relative_url}}' width="70%"></center>

#### Ablation of PoseVAE
다음은 PoseVAE의 다양성과 오디오 정렬에 대한 ablation 결과이다. 

<center><img src='{{"/assets/img/sadtalker/sadtalker-table4.PNG" | relative_url}}' width="48%"></center>

#### Ablation of Face Render
다음은 Face Render에 대한 ablation 결과이다. 

<center><img src='{{"/assets/img/sadtalker/sadtalker-fig9.PNG" | relative_url}}' width="70%"></center>

### 4. Limitation
<center><img src='{{"/assets/img/sadtalker/sadtalker-fig10.PNG" | relative_url}}' width="70%"></center>
<br>
3DMM은 눈과 치아의 변형을 모델링하지 않기 때문에 Face Render의 mappingNet은 경우에 따라 실제 치아를 합성하는 데 어려움을 겪는다. 이러한 한계는 위 그림과 같이 얼굴 복원 네트워크 (GFPGAN)를 통해 개선될 수 있다. 또 다른 한계는 감정과 시선 방향과 같은 다른 얼굴 표정 외에 입술 동작과 눈 깜박임에만 관심이 있다는 것이다. 따라서 생성된 영상은 고정된 감정을 가지게 되고, 이로 인해 생성된 콘텐츠의 현실감도 떨어지게 된다. 