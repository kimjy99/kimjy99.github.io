---
title: "[논문리뷰] Reflective Gaussian Splatting"
last_modified_at: 2025-03-06
categories:
  - 논문리뷰
tags:
  - Gaussian Splatting
  - Novel View Synthesis
  - 3D Vision
  - ICLR
excerpt: "Ref-Gaussian 논문 리뷰 (ICLR 2025)"
use_math: true
classes: wide
---

> ICLR 2025. [[Paper](https://arxiv.org/abs/2412.19282)] [[Page](https://fudan-zvg.github.io/ref-gaussian/)] [[Github](https://github.com/fudan-zvg/ref-gaussian)]  
> Yuxuan Yao, Zixuan Zeng, Chun Gu, Xiatian Zhu, Li Zhang  
> Fudan University | ByteDance  
> 26 Dec 2024  

<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-fig1.webp" | relative_url}}' width="100%"></center>

## Introduction
대부분의 [3DGS](https://kimjy99.github.io/논문리뷰/3d-gaussian-splatting)는 디테일한 반사 성분을 캡처할 수 없는 본질적인 한계로 인해 반사 표면을 모델링하는 데 어려움을 겪는다. 각 Gaussian에서 단순화된 shading function을 사용하면 빠른 수렴을 달성할 수 있지만 형상, 재료, 조명을 모델링할 때 상당한 노이즈가 발생한다. 이 문제를 피하기 위해 알파 블렌딩 후 픽셀 레벨에서 shading function을 수행하는 deferred shading을 사용하면 gradient를 부드럽게 하는 데 도움이 된다. 하지만 단순화된 shading function을 사용하면 복잡한 구조와 표면 및 inter-reflection이 있는 복잡한 반사 물체를 모델링하는 데 효과적이지 않다. 몬테카를로 샘플링을 통해 보다 정확한 shading을 수행하면 추가적인 계산 오버헤드가 발생하며, visibility를 계산하기 위해 Gaussian에 대한 ray tracing이 필요하다. 

위에서 언급한 모든 문제를 해결하기 위해, 본 논문은 **Reflective Gaussian splatting (Ref-Gaussian)** 프레임워크를 제안하여 복잡한 inter-reflection 효과를 고려하면서 반사 물체를 실시간으로 고품질로 렌더링한다. Ref-Gaussian은 두 가지 핵심 구성 요소로 구성된다. 

1. **Physically based deferred rendering**: 픽셀 레벨의 재료 속성(BRDF)으로 렌더링 방정식을 계산하고, split-sum approximation으로 몬테카를로 샘플링의 무거운 계산을 피한다. 
2. **Gaussian grounded inter-reflection**: 메쉬를 추출하여 Gaussian splatting에 대한 복잡한 inter-reflection를 계산한다. 

저자들은 형상 모델링을 강화하기 위해, 장면 표현으로 2D Gaussian을 선택하고, 초기 Gaussian shading 단계와 material-aware normal propagation process를 도입했다.

## Method
<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-fig2.webp" | relative_url}}' width="100%"></center>

### 1. Physically based deferred rendering
Physically based deferred rendering은 Disney BRDF 모델의 단순화된 버전을 사용한다. 구체적으로, 각 Gaussian은 재료 관련 속성 albedo $\lambda \in [0,1]^3$, metallic $m \in [0,1]$, roughness $r \in [0,1]$과 연관된다. Normal 벡터 $n \in [0,1]^3$은 $n = t_u \times t_v$로 계산할 수 있다. 알파 블렌딩을 통해 얻은 feature map에 렌더링 방정식을 적용한다.

$$
\begin{equation}
X = \sum_{i=1}^N x_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j), \quad \textrm{where} \; X = \begin{bmatrix} \Lambda \\ M \\ R \\ N \end{bmatrix}, \; x_i = \begin{bmatrix} \lambda_i \\ m_i \\ r_i \\ n_i \end{bmatrix}
\end{equation}
$$

알파 블렌딩 전에 Gaussian에 직접 셰이딩하는 것과 달리, deferred shading은 알파 블렌딩을 smoothing filter로 취급한다. 이를 통해 feature의 최적화를 안정화하고 최종적으로 더욱 응집력 있는 렌더링 결과를 생성할 수 있다.

Albedo $\Lambda$, metallic $M$, roughness $R$, normal $N$을 포함한 material map을 집계한 후, 이를 사용하여 렌더링 방정식을 적용한다. 렌더링 방정식은 $$\omega_o$$ 방향으로 나가는 outgoing radiance $$L(\omega_o)$$를 다음과 같이 표현한다.

$$
\begin{equation}
L(\omega_o) = \int_\Omega L_i (\omega_i) f(\omega_i, \omega_o) (\omega_i \cdot N) d \omega_i
\end{equation}
$$

여기서 $f(\omega_i, \omega_o)$는 BRDF이며, 반구에 반사된 입사광의 적분이다. 

BRDF는 diffuse 항과 specular 항으로 구성되므로 적분도 두 부분으로 나눌 수 있다. Diffuse 항은 normal $N$을 사용하여 미리 적분된 environment map을 쿼리하고 재료 항과 곱하여 계산하는 반면, specular 항은 표면 반사를 설명하기 위해 더 복잡한 반사 계산이 필요하다. 여기서 BRDF의 specular 항은 다음과 같다.

$$
\begin{equation}
f_s (\omega_i, \omega_o) = \frac{D \, G \, F}{4 (\omega_o \cdot N) (\omega_i \cdot N)}
\end{equation}
$$

($D$는 GGX normal distribution function, $F$는 Fresnel term, $G$는 shadowing-masking term)

위의 적분 항을 계산하기 위해 일반적인 접근 방식은 몬테카를로 샘플링이지만, 실시간으로 렌더링하는 데는 계산 비용이 많이 든다. 대신 [split-sum approximation](https://arxiv.org/abs/2111.12503)을 채택한다.

$$
\begin{equation}
L_s (\omega_o) \approx \int_\Omega f_s (\omega_i, \omega_o) (\omega_i \cdot N) d \omega_i \cdot \int_\Omega L_i (\omega_i) D (\omega_i, \omega_o) (\omega_i \cdot N) d \omega_i
\end{equation}
$$

여기서 첫 번째 항은 $$(\omega_i \cdot N)$$과 roughness $R$에만 의존하므로 결과를 미리 계산하여 2D 텍스처 맵에 저장할 수 있다. 두 번째 항은 specular lobe에 대한 입사 radiance의 적분을 나타내며, 각 학습 iteration의 시작 부분에서 여러 roughness 레벨에 대해 사전 통합할 수 있다. 이를 통해 반사 방향과 roughness를 파라미터로 사용하여 trilinear interpolation을 수행하여 환경 조명(environment lighting)을 나타내는 일련의 큐브맵을 효율적으로 사용할 수 있다. 

두 번째 항은 직접광 $$L_\textrm{dir}$$에 해당한다. 

### 2. Gaussian grounded inter-reflection
Inter-reflection은 반사 물체를 렌더링하는 데 중요한 역할을 한다. 이를 위해 위 식의 두 번째 항을 직접광 $$L_\textrm{dir}$$과 간접광 $$L_\textrm{ind}$$으로 별도로 모델링하여 specular 성분을 더욱 향상시킨다. 직접광 $$L_\textrm{dir}$$은 반사가 어떤 장면 요소에도 차단되지 않는 것을 나타내고 나머지는 간접광 $$L_\textrm{ind}$$로 정의된다.

구체적으로, 반사 방향 $$R = 2(\omega_o \cdot N) N − \omega_o$$를 따라 물체가 스스로 가려지는 지 여부에 따라 입사광의 visibility $$V \in \{0, 1\}$$를 근사한다. 가려진 부분에서 나오는 간접광을 $$L_\textrm{ind}$$로 표현한다.

$$
\begin{equation}
L_s^\prime \approx (\int_\Omega f_s (\omega_i, \omega_o) (\omega_i \cdot N) d \omega_i) \cdot [L_\textrm{dir} \cdot V + L_\textrm{ind} \cdot (1 - V)]
\end{equation}
$$

직관적으로, $$L_\textrm{ind}$$은 환경 조명 추정 시에 가려짐으로 인해 발생하는 교란을 효과적으로 모델링하는 것을 목표로 한다.

Gaussian을 가로지르는 광선을 따라 visibility를 효율적으로 계산하는 것은 어려운 일이다. 저자들은 추출된 메쉬에 대해 즉석에서 ray tracing을 수행하였다. 최적화하는 동안, TSDF fusion을 사용하여 주기적으로 물체의 표면 메쉬를 추출한다. 메쉬가 구성되면 ray tracing을 사용하여 광선과 표면 사이의 교차점을 계산하여 각 픽셀이 가려졌는지 확인한다. 효율성을 위해 bounding volume hierarchy (BVH)를 사용하여 ray tracing을 가속화한다. 

간접광 성분의 경우 각 Gaussian에 spherical harmonics (SH)로 모델링된 뷰에 따른 색상 $$l_\textrm{ind}$$로 할당된다. 렌더링 프로세스 동안 $$l_\textrm{ind}$$는 Gaussian 레벨에서 반사 방향으로 평가되고 알파 블렌딩이 적용되어 다음과 같이 간접광 map으로 집계된다.

$$
\begin{equation}
L_\textrm{ind} = \sum_{i=1}^N l_\textrm{ind} \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)
\end{equation}
$$

### 3. Geometry focused model optimization
##### Initial stage with per-Gaussian shading
픽셀 레벨 셰이딩은 shading function이 형상의 gradient 방향을 교란하기 때문에 초기 단계에서 수렴 문제를 야기한다. 저자들은 이를 해결하기 위해 렌더링 방정식을 각 Gaussian에 직접 적용하여 이와 연관된 재료 및 형상 속성을 사용하여 outgoing radiance $$L(\omega_o)$$를 계산하는 초기 단계를 제안하였다. 그런 다음 $$L(\omega_o)$$는 rasterization 중에 알파 블렌딩되어 최종 PBR 렌더링을 생성한다. 이 Gaussian 레벨 셰이딩 디자인은 gradient를 Gaussian으로 다시 효과적으로 전송하고 궁극적으로 최적화 프로세스를 용이하게 하는 데 도움이 될 수 있다.

##### Material-aware normal propagation
부정확한 normal이 있는 위치는 specular 성분을 포착하는 데 어려움을 겪는 경우가 많다. Normal 정확도는 높은 metallic, 낮은 roughness 속성과 강력한 양의 상관 관계가 있다. 이러한 통찰력에 따라 높은 metallic과 낮은 roughness를 가진 2D Gaussian의 scale을 주기적으로 늘려서 normal 정보를 인접한 Gaussian으로 전파하여 더 정확한 형상을 포착할 수 있도록 한다.

##### Loss function
전체 loss function은 3DGS의 reconstruction RGB loss $L_c$, normal consistency loss $L_n$, edge-aware normal smoothness loss $$L_\textrm{smooth}$$로 구성된다. 

$$
\begin{equation}
L = L_c + \lambda_n L_n + \lambda_\textrm{smooth} L_\textrm{smooth} \\
\textrm{where} \quad L_n = 1 - \tilde{N}^\top N, \; L_\textrm{smooth} = \| \nabla N \| \exp (-\| \nabla C_\textrm{gt} \|)
\end{equation}
$$

($\tilde{N}$은 depth map으로부터 계산한 normal)

## Experiments
- 데이터셋: Shiny Blender, Glossy Synthetic, Ref-Real
- 학습 디테일
  - 1단계: per-Gaussian rendering
    - iteration: 1.8만
  - 2단계: deferred rendering
    - 1단계의 색상과 재료 속성(albedo, metallic, roughness)은 초기화하고 형상만 유지
    - iteration: 4만
  - learning rate: 재료 속성들은 0.005, environment map은 0.01
  - material-aware normal propagation 조건: metallic이 0.02 이상, roughness가 0.1 이하
  - $$\lambda_n$$ = 0.05, $$\lambda_\textrm{smooth}$$ = 1.0
  - 메쉬 추출은 3천 iteration마다
  - GPU: NVIDIA A6000 1개

### 1. Comparisons
다음은 장면별로 렌더링 품질을 비교한 결과이다. 

<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-table1.webp" | relative_url}}' width="100%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-fig3.webp" | relative_url}}' width="95%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><div style="overflow-x: auto; width: 77%;">
  <div style="width: 200%;">
    <img src='{{"/assets/img/ref-gaussian/ref-gaussian-fig4.webp" | relative_url}}' width="100%">
  </div>
</div></center>
<br>
다음은 normal map 품질, environment map 품질, 학습 시간, 렌더링 속도를 비교한 표이다. 

<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-table2.webp" | relative_url}}' width="75%"></center>
<br>
다음은 inverse rendering 결과이다. 

<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-fig5.webp" | relative_url}}' width="100%"></center>
<br>
다음은 예측된 environment map을 비교한 것이다. 

<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-fig6.webp" | relative_url}}' width="100%"></center>

### 2. Ablation Study
다음은 Ref-Gaussian의 각 구성 요소에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-table4.webp" | relative_url}}' width="94%"></center>
<br>
다음은 형상 최적화에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-table3.webp" | relative_url}}' width="23%"></center>
<br>
다음은 physically based rendering (PBR)에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-fig7.webp" | relative_url}}' width="42%"></center>
<br>
다음은 2DGS 사용에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-fig8.webp" | relative_url}}' width="57%"></center>
<br>
다음은 inter-reflection에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-fig9.webp" | relative_url}}' width="70%"></center>
<br>
다음은 material-aware normal propagation에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/ref-gaussian/ref-gaussian-fig10.webp" | relative_url}}' width="70%"></center>