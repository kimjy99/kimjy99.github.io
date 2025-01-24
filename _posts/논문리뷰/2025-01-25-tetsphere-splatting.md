---
title: "[논문리뷰] TetSphere Splatting: Representing High-Quality Geometry with Lagrangian Volumetric Meshes"
last_modified_at: 2025-01-25
categories:
  - 논문리뷰
tags:
  - 3D Vision
  - Novel View Synthesis
  - AI
excerpt: "TetSphere Splatting 논문 리뷰"
use_math: true
classes: wide
---

> arXiv 2024. [[Paper](https://arxiv.org/abs/2405.20283)] [[Github](https://github.com/gmh14/tssplat)]  
> Minghao Guo, Bohan Wang, Kaiming He, Wojciech Matusik  
> MIT CSAIL  
> 30 May 2024  

<center><img src='{{"/assets/img/tetsphere-splatting/tetsphere-splatting-fig1.webp" | relative_url}}' width="95%"></center>

## Introduction
3D 모델링 발전의 중심에는 형상 표현 방식이 있으며, 크게 Eulerian 표현과 Lagrangian 표현으로 구분될 수 있다. Eulerian 표현은 사전에 정의된 3D 공간의 고정된 좌표 세트에서 형상을 설명하며, 각 좌표 위치는 볼륨 내부 점유 여부나 표면에서의 거리와 같은 속성과 연관된다. 널리 사용되는 Eulerian 표현으로는 연속적인 공간 좌표를 입력으로 받아 density field나 SDF를 모델링하는 신경망, 그리고 grid vertex들에서 signed distance 값을 정의하는 deformable grid가 있다.

Eulerian 표현은 그 인기에 비해 계산 복잡도와 품질 사이의 trade-off라는 문제에 직면한다. 형상의 세밀한 디테일을 포착하려면 높은 용량의 신경망 또는 고해상도의 grid가 필요하며, 이는 시간과 메모리 측면에서 최적화 비용이 매우 높다. 이러한 trade-off는 종종 Eulerian 표현이 얇고 가느다란 구조를 모델링하는 데 제약을 초래한다. 이는 사전에 정의된 해상도가 세밀한 디테일을 포착하기에 충분하지 않기 때문이다.

최근에는 Lagrangian 표현 방식으로의 전환이 점점 증가하고 있으며, 이는 Eulerian 방식보다 일반적으로 계산 효율성이 더 높다. Lagrangian 표현은 3D 공간에서 primitive의 움직임을 추적하여 3D 형상을 설명한다. 이러한 primitive들은 형상의 local geometry에 따라 적응적으로 배치될 수 있어, 특히 세밀한 기하학적 디테일을 모델링할 때 Eulerian 방식에 비해 적은 계산 자원을 요구한다. 가장 널리 사용되는 Lagrangian primitive로는 3D Gaussian과 surface triangle이 있다. 

Lagrangian 표현은 계산 효율성 측면에서 선호되지만, 종종 메쉬 품질이 낮다는 문제가 있다. 이는 개별 점 또는 삼각형을 추적하는 데 의존하기 때문에 전체적인 구조적 일관성이 부족해질 수 있기 때문이다. 예를 들어, 3D Gaussian은 공간에서 자유롭게 움직일 수 있어 종종 노이즈가 많은 메쉬를 생성할 수 있고, surface triangle은 non-manifold 표면을 형성하거나 불규칙하고 때로는 degenerated triangle을 생성할 수 있다. 이러한 기하학적 문제를 포함한 결과물은 고품질 메쉬가 필수적인 렌더링이나 시뮬레이션에 적합하지 않다. 

이러한 문제를 해결하기 위해, 본 논문은 고품질 메쉬를 생성하는 데 중점을 둔 새로운 Lagrangian 표현 방식인 **TetSphere Splatting**을 제안하였다. 저자들의 주요 통찰은 기존의 Lagrangian primitive가 지나치게 세밀하여 고품질 메쉬를 보장하기 어렵다는 점에서 비롯된다. 메쉬 품질은 개별 primitive뿐만 아니라 그들 간의 상호작용에도 크게 의존한다. 예를 들어, primitive들이 적절하게 정렬되어 있는지, primitive가 얼마나 잘 연결되어 있는지에 따라 달라진다. 

본 논문의 표현 방식은 primitive로 볼륨 기반의 사면체 구를 사용하며, 이를 **TetSphere**라고 명명한다. 기존의 개별 점이나 삼각형과 달리, TetSphere는 사면체화(tetrahedralization)를 통해 연결된 점 집합으로 구성된 구이다. 초기에는 균일한 구 형태로 시작하여, 각 TetSphere는 복잡한 형태로 변형될 수 있다. 변형된 TetSphere의 집합이 모여 3D 형상을 나타내며, 이는 Lagrangian 접근 방식과 부합한다.

이 구조화된 primitive는 기하학적 정규화와 제약 조건을 각 TetSphere 내의 점들 사이에 적용할 수 있게 하여, 변형 과정에서도 메쉬 품질을 유지할 수 있도록 한다. TetSphere의 volumetric한 특성은 볼륨 전반에 걸쳐 점들이 응집력 있는 배열을 이루도록 하며, 불규칙한 삼각형이나 non-manifoldness와 같은 일반적인 표면 메쉬 문제를 효과적으로 줄인다. 이러한 접근 방식은 구조적 무결성을 보장하며, 고품질 메쉬 생성을 가능하게 한다.

본 논문은 TetSphere splatting을 위한 계산 프레임워크도 제안하였다. [Gaussian splatting](https://kimjy99.github.io/논문리뷰/3d-gaussian-splatting)과 유사하게, TetSphere를 splatting하여 목표 형상에 부합하도록 만든다. TetSphere의 변형을 기하학적 에너지 최적화 문제로 정식화하며, 여기에는 미분 가능한 렌더링 loss, deformation gradient field의 bi-harmonic energy, 그리고 non-inversion 제약 조건들이 포함된다. 이 모든 요소는 gradient descent를 통해 효과적으로 해결할 수 있다.

SOTA 방법들과 비교했을 때, TetSphere splatting은 다른 지표에서 경쟁력 있는 성능을 유지하면서도 우수한 메쉬 품질을 보여준다. 또한, 단일 이미지 또는 텍스트로부터 3D 형상을 생성할 수 있다. 

## TetSphere Splatting
<center><img src='{{"/assets/img/tetsphere-splatting/tetsphere-splatting-fig3.webp" | relative_url}}' width="60%"></center>
<br>
저자들은 primitive로 tetrahedral sphere를 선택하였다. Point cloud와 달리, 사면체 메쉬들은 사면체화(tetrahedralization)로 인해 점들 간의 구조화된 local connectivity를 강제한다. 이는 3D 형상의 기하학적 무결성을 보존할 뿐만 아니라, 메쉬 내부 전체에 걸쳐 기하학적 정규화를 적용함으로써 표면 품질을 향상시킨다.

저자들은 TetSphere splatting을 통해 형상 재구성을 TetSphere의 변형(deformation)으로 공식화하였다. 초기 상태에서는 TetSphere 집합을 사용하며, 이들의 vertex 위치들을 조정하여 해당 메쉬의 렌더링 이미지가 타겟 멀티뷰 이미지와 정렬되도록 한다. Vertex의 이동은 geometry processing 분야에서 도출된 두 가지 기하학적 정규화로 제약된다.

1. **Bi-harmonic energy**를 통해 매끄럽지 않은 변형을 억제
2. **Local injectivity**를 통해 메쉬가 변형된 후 뒤집히는 것, 즉 inversion을 방지

이러한 정규화는 생성된 사면체 메쉬가 우수한 품질을 유지하고 구조적 무결성을 보장하는 데 효과적인 것으로 입증되었다. 

### 1. Tetrahedral Sphere Primitive
TetSphere splatting의 primitive는 $N$개의 vertex와 $T$개의 사면체를 갖는 TetSphere라고 하는 사면체화된 구이다. 유한 요소법(FEM)의 원리를 적용하여 각 TetSphere의 메쉬는 사면체 요소로 구성된다. 

$i$번째 변형된 메쉬의 모든 정점의 위치 벡터를 $x_i \in \mathbb{R}^{3 \times N}$으로 표시한다. $i$번째 구에서 $j$번째 사면체의 deformation gradient $$\textbf{F}_\textbf{x}^{(i,j)} \in \mathbb{R}^{3 \times 3}$$는 각 사면체의 모양이 어떻게 변형되는지 정량적으로 설명한다. 본질적으로 deformation gradient $$\textbf{F}_\textbf{x}^{(i,j)}$$는 사면체가 원래 구성에서 변형된 상태로 겪는 공간적 변화를 측정하는 역할을 한다. 

하나의 구를 사용하는 대신, 임의의 형태를 정확하게 표현하기 위해 여러 개의 구 집합을 사용한다. 따라서 전체 형태는 모든 구의 합집합으로 나타난다. 여러 구를 채택함으로써, 각 지역을 독립적으로 자세하게 표현할 수 있어 매우 정밀한 표현이 가능하다. 또한, 임의의 위상을 가진 형태도 표현할 수 있게 해준다. 

<center><img src='{{"/assets/img/tetsphere-splatting/tetsphere-splatting-fig2.webp" | relative_url}}' width="100%"></center>
<br>
TetSphere를 사용하는 것은 재구성을 위한 기존 표현 방식과 비교할 때 여러 가지 기술적 이점을 제공한다. 

- vs. 뉴럴 표현 방식 (ex. [NeRF](https://kimjy99.github.io/논문리뷰/nerf)): 신경망에 의존하지 않기 때문에 최적화 과정을 본질적으로 가속화할 수 있다.
- vs. Eulerian 표현 방식 (ex. [DMTet](https://arxiv.org/abs/2111.04276)): Iso-surface 추출이 전혀 필요하지 않다. 이는 grid의 미리 정해진 해상도로 인해 메쉬 품질이 저하되는 문제를 피할 수 있다.
- vs. 다른 Lagrangian 표현 방식: 각 사면체는 vertex들 간의 제약 조건을 부여하여, 우수한 메쉬 품질을 보장한다.

### 2. TetSphere Splatting as Shape Deformation
<center><img src='{{"/assets/img/tetsphere-splatting/tetsphere-splatting-fig4.webp" | relative_url}}' width="100%"></center>
<br>
대상 물체의 형상을 재구성하기 위해 vertex 위치를 변경하여 초기 TetSphere들을 변형한다. 이 프로세스에는 두 가지 주요 목표가 있다. 

1. 변형된 TetSphere들이 입력된 멀티뷰 이미지와 일치해야 한다. 
2. 필요한 제약 조건을 준수하는 높은 메쉬 품질을 유지해야 한다. 

메쉬 품질을 유지하기 위해, deformation gradient field에 bi-harmonic energy를 적용한다. Bi-harmonic energy는 geometry processing 분야에서 필드 전체의 부드러움을 정량화하는 에너지로 정의된다. 이러한 기하학적 정규화는 변형 과정에서 deformation gradient field의 부드러움을 보장하여, 불규칙한 메쉬나 울퉁불퉁한 표면을 방지한다. 

중요한 점은 bi-harmonic 정규화가 최종 결과의 과도한 부드러움을 초래하지 않는다는 것이다. 그 이유는 에너지가 절대적인 vertex 위치가 아니라 vertex 위치의 상대적 변화를 측정하는 deformation gradient field를 대상으로 하기 때문이다. 이러한 접근법은 물리 시뮬레이션에서 사용하는 기법과 유사하게, 날카로운 로컬 디테일을 보존할 수 있게 해준다.

또한, 모든 변형된 요소에서 local injectivity를 보장하기 위한 기하학적 제약 조건을 도입한다. 이를 통해 요소들이 변형 중에 방향을 유지하도록 보장하며, 뒤집히거나 안팎이 바뀌는 것을 방지한다. 이 제약 조건은 수학적으로 다음과 같이 표현될 수 있다.

$$
\begin{equation}
\textrm{det} (\textbf{F}_\textbf{x}^{(i,j)}) > 0
\end{equation}
$$
 
중요한 점은 부드러움을 위한 bi-harmonic energy와 요소 방향을 위한 local injectivity라는 두 가지 조건이 모두 geometry processing에 기반을 두고 있기 때문에, 어떠한 사면체 메쉬에도 보편적으로 적용할 수 있다는 것이다.

TetSpheres의 변형은 다음과 같은 최적화 문제로 공식화된다.

$$
\begin{aligned}
& \min_{\textbf{x}} \; \Phi (R(\textbf{x})) + \| \textbf{LF}_\textbf{x} \|_2^2 \\
& \textrm{s.t.} \; \textrm{det} (\textbf{F}_\textbf{x}^{(i,j)}) > 0, \; \forall i \in \{1, \ldots, M\}, \; j \in \{1, \ldots, T\}
\end{aligned}
$$

- $\textbf{x} = [x_1, \ldots, x_M] \in \mathbb{R}^{3NM}$: 모든 $M$개의 TetSphere에 걸쳐 있는 vertex들의 위치
- $$\textbf{F}_\textbf{x} \in \mathbb{R}^{9MT} = [\textrm{vec}(\textbf{F}_\textbf{x}^{(1,1)}), \ldots, \textrm{vec}(\textbf{F}_\textbf{x}^{(M,T)})]$$: 모든 TetSphere의 flatten된 deformation gradient field
- $\textbf{L} \in \mathbb{R}^{9MT \times 9MT}$: 사면체 면의 연결성(connectivity)에 기반하여 정의되는 bi-harmonic energy의 Laplacian matrix
  - 각 block $$\textbf{L}_{pq} \in \mathbb{R}^{9 \times 9}, p \ne q$$는 $p$번째와 $q$번째 사면체가 공통된 삼각형을 공유하는 경우 $-I$로 설정됨
  - $$\textbf{L}_{pp}$$의 경우는 $p$번째 사면체의 이웃 개수인 $k$에 비례하여 $kI$로 설정됨
  - block symmetric: $$\textbf{L}_{pq} = \textbf{L}_{qp}$$
- $R (\cdot)$: 렌더링 함수
- $\Phi (\cdot)$: 변형된 TetSphere의 합집합과 입력 이미지를 매칭하는 렌더링 loss

이 제약 조건이 있는 최적화를 쉽게 다루기 위해, 제약 조건을 목적함수에 통합하여 위 식을 재구성한다. 

$$
\begin{equation}
\min_{\textbf{x}} \; \Phi (R(\textbf{x})) + w_1 \| \textbf{LF}_\textbf{x} \|_2^2 + w_2 \sum_{i,j} (\min \{0, \textrm{det} (\textbf{F}_\textbf{x}^{(i,j)})\})^2
\end{equation}
$$

위 식은 gradient descent를 통한 최적화가 가능하다.

제안된 최적화 프레임워크에서는 다음 세 가지를 고려하고 있다. 

1. $\Phi (\cdot)$는 컬러 이미지의 $$\ell_1$$, 깊이 이미지의 MSE, normal 이미지의 cosine embedding loss를 포함한다. 
2. 사면체가 선형적인 요소이므로 deformation gradient $$\textbf{F}_\textbf{x}^{(i,j)}$$는 $\textbf{x}$의 선형 함수이고, 이는 bi-harmonic energy를 이차항으로 만든다.
3. 가중치 $w_1$과 $w_2$는 cosine scheduler를 사용하여 동적으로 조정된다.

## TetSphere Initialization and Texture Optimization
#### TetSphere 초기화
멀티뷰 이미지를 입력으로 받아, TetSphere의 3D 중심 위치를 초기화하기 위해 feature point들을 선택한다. 이 과정에서 물체 내부에 이러한 TetSphere들이 균일하게 분포하도록 하여, 멀티뷰 이미지에 나타난 실루엣을 포괄적으로 커버하려고 한다.

임의의 형태에 대해 TetSphere의 초기 중심을 자동으로 선택하기 위해 **silhouette coverage**라는 알고리즘을 도입하며, 이는 [Coverage Axis](https://arxiv.org/abs/2110.00965)에서 영감을 받았다.

1. Coarse한 voxel grid를 구축하고 각 voxel에 처음에는 0을 할당한다.
2. 입력 멀티뷰 이미지와 동일한 카메라 포즈를 사용하여 이 voxel을 image space로 projection한다. 
3. 모든 이미지의 전경에 속한 voxel은 값이 1로 표시된다. 이러한 voxel 위치는 TetSphere 중심의 후보 위치가 된다. 
4. 모든 후보 위치에 다양한 반지름 값을 가진 균일한 구를 배치하고, 모든 후보 위치를 포괄적으로 덮는 최소한의 집합을 선택한다. 

저자들은 이 선택을 효율적으로 수행하기 위해 linear programming problem로 문제를 공식화했다. 실제 구현에서 voxel grid의 해상도는 300$\times$300이고 $n = 20$인 경우, 전체 TetSphere 초기화는 평균적으로 약 1분 이내에 완료된다. 

#### 텍스처 최적화
TetSphere splatting의 주요 목적은 고품질의 형상 표현을 제공하는 것이지만, 그 명시적인 구조 덕분에 텍스처와 재료를 TetSphere의 표면 vertex와 면에 직접 적용할 수 있다. TetSphere splatting의 중요한 장점 중 하나는 사면체 구의 변형이 표면의 위상(topology)을 보존한다는 점으로, 이를 통해 Disney의 principled BRDF와 같은 고급 재료 모델을 물리 기반 렌더링에 매끄럽게 통합할 수 있다. 이는 텍스트를 3D로 변환하는 생성과 같은 응용 분야에서 유용하다. 하지만 텍스처 최적화는 선택적인 기능이며, 비교 대상으로 다루지는 않았다. 

## Experiments
- Metric
  - Chamfer Distance (Cham.), Volume IoU (Vol. IoU)
  - F-Score, Normal Consistency, Edge Chamfer Distance, Edge F-Score
  - Area-Length Ratio (ALR): 표면 메쉬 내에서 삼각형의 면적과 둘레의 평균 비율
  - Manifoldness Rate (MR): 닫힌 다면체(Manifold)의 비율
  - Connected Component Discrepancy (CC Diff.): 메쉬 내에 floater나 구조적 불연속성의 존재를 식별

### 1. Results
다음은 멀티뷰 재구성 결과를 비교한 것이다.

<center><img src='{{"/assets/img/tetsphere-splatting/tetsphere-splatting-fig5.webp" | relative_url}}' width="100%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/tetsphere-splatting/tetsphere-splatting-table1.webp" | relative_url}}' width="78%"></center>
<br>
다음은 GSO 데이터셋에서 단일 뷰 재구성 결과를 비교한 표이다. 

<center><img src='{{"/assets/img/tetsphere-splatting/tetsphere-splatting-table2.webp" | relative_url}}' width="78%"></center>

### 2. Analysis
다음은 메모리 비용과 실행 속도를 image-to-3D 생성 방법들과 비교한 표이다. (GPU: 40GB A100)

<center><img src='{{"/assets/img/tetsphere-splatting/tetsphere-splatting-table3.webp" | relative_url}}' width="43%"></center>
<br>
다음은 가중치 $w_1$과 $w_2$에 대한 분석 결과이다. 

<center><img src='{{"/assets/img/tetsphere-splatting/tetsphere-splatting-fig6.webp" | relative_url}}' width="60%"></center>
<br>
다음은 text-to-3D 생성 결과를 [RichDreamer](https://arxiv.org/abs/2311.16918)와 비교한 것이다. 

<center><img src='{{"/assets/img/tetsphere-splatting/tetsphere-splatting-fig7.webp" | relative_url}}' width="90%"></center>