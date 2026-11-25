---
title: "[논문리뷰] Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"
last_modified_at: 2026-11-26
categories:
  - 논문리뷰
tags:
  - LLM
excerpt: "Weight-Space Programming 논문 리뷰"
use_math: true
classes: wide
---

> [[Blog](https://rikiyat.github.io/blog/weight-space-programming/)] [[Github](https://github.com/RikiyaT/lm_understand-logs)]  
> Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab  
> MIT CSAIL  
> 2 Nov 2026  

<center><img src='{{"/assets/img/weight-space-programming/weight-space-programming-fig1.webp" | relative_url}}' width="83%"></center>

## Introduction
오늘날 신경망의 가중치는 최적화 과정을 통해 생성되거나, 가중치를 생성하도록 학습된 별도의 신경망을 통해 만들어지며, 혹은 정형화된 프로그램으로부터 컴파일러를 거쳐 생성되기도 한다. 본 논문에서는 자연어를 기반으로 작동하는 최신 LLM들이, 자연어로 주어진 목표에 맞춰 신경망 가중치를 이해하고 생성하는 데 있어 놀라울 정도로 뛰어난 능력을 보이기 시작했다는 점을 살펴보았다.

지난 몇 년간 사람들은 프롬프트나 이를 기반으로 구축된 프로그램 같은 텍스트 결과물을 최적화하기 위해 LLM을 활용해 왔다. 최근 들어서는 머신러닝 모델을 학습시키기 위한 전체 학습 환경은 물론, 학습 코드와 hyperparameter까지 최적화하는 데 LLM을 사용하기 시작했다.

종종 이러한 방법들을 정당화할 때, 미분 가능한 신경망의 가중치를 업데이트하는 경우와 달리 텍스트 결과물을 대상으로 성능을 점진적으로 개선할 수 있는 다른 원칙적인 방법을 알지 못한다는 점을 근거로 들곤 한다. 예를 들어, 인간은 직접 코드를 수정하여 최적화할 수 있으므로 LLM에게도 그러한 작업을 맡기는 것은 타당해 보였다. 반면, 인간이 직접 신경망의 가중치를 최적화할 수는 없기에, 해당 영역은 대체로 symbolic algorithm의 영역으로 남아 있었다.

본 논문에서는 추론 기반 최적화(reasoning-driven optimization)라는 패러다임이 이제 미분 가능한 신경망 영역에도 도입되고 있음을 시사하는, 매우 초기 단계의 연구 결과를 공유한다.

**Weight-Space Programming**은 신경망의 가중치를 마치 프로그래밍 언어로 작성된 프로그램처럼 다루는 개념이다. 이는 가중치를 단순히 최적화 알고리즘에 전적으로 맡겨두는 대신, 직접 가중치를 읽어 신경망이 무엇을 계산하는지 파악하고, 이를 직접 수정하거나 편집함으로써 신경망의 계산 내용을 바꾸는 방식이다. 이 접근 방식에서 프로그래머 역할을 수행하는 것은 범용 LLM이다. LLM은 gradient 정보 없이 자연어로 된 spec만을 바탕으로 작업하며, 때로는 환경으로부터 어떠한 피드백도 받지 않은 상태에서 가중치를 다루기도 한다.

자연어를 사용하여 LLM에게 그냥 물어보는 것만으로, 신경망 가중치 내부에 숨겨진 특성이나 결함을 찾아내고, 해당 결함을 수정하거나, 심지어 처음부터 새로운 신경망을 구축하도록 하는 것이 가능해지고 있다. 저자들은 소규모 신경망에서 시작하여, 실제 언어 모델에 이르기까지의 다음과 같은 성과를 소개하였다.

1. LLM은 신경망 가중치를 읽어낼 수 있다. (특정 신경망이 어떤 게임을 하도록 학습되었는지, diffusion model이 어떤 형태를 그려내는지, 소규모 언어 모델이 어떤 언어로 학습되었는지, 두 언어 모델 중 어느 것이 아첨하는 성향을 갖도록 학습되었는지)
2. LLM은 신경망 가중치를 수정할 수도 있다. (Physical control task를 제대로 수행하지 못하는 신경망을 수정하기, Qwen3.5-0.8B 모델의 수학 능력을 향상하기)
3. LLM은 신경망 가중치를 처음부터 새로 작성할 수도 있다. (세 가지 MinAtar 게임을 수행할 수 있는 신경망, 바둑을 둘 수 있는 신경망)

## Experiments
### 1. Can LLMs understand model behaviors from weights?
##### Astra는 신경망의 가중치만 보고도 해당 신경망이 어떤 게임을 학습했는지 알아낼 수 있을까?
저자들은 PPO를 사용하여 네 가지 컴퓨터 게임(Breakout, Pong, Pac-Man, Freeway)을 플레이하도록 신경망 policy를 학습시켰다. 네 신경망 모두 동일한 입력 채널을 사용하고, 동일한 action들 중에서 선택하며, 신경망 구조의 크기 또한 완전히 동일하다. 따라서 가중치를 직접 확인하지 않고는 어떤 게임을 학습했는지 알 수 없는 상태이다.

저자들은 각 신경망의 가중치 데이터를 프롬프트 형태로 Astra에 입력하고, 이 신경망은 어떤 게임을 학습했는지 질문했다. 고려 대상인 게임 목록을 전혀 알려주지 않았음에도, Astra는 네 신경망 모두에 대해 정확한 게임 이름을 맞혔다. Astra가 선택할 수 있는 게임의 가짓수가 매우 많기 때문에 이는 매우 흥미로운 결과이다.

> Channel 0 — agent: Strong weights are concentrated on row 8, across the playable columns.  
> At (8,1), weights include 4.066 to hidden unit 15 and −4.013 to unit 31.  
> This suggests a bottom-mounted paddle moving horizontally.  
> Channel 1 — moving objects: Large, structured weights cover the interior above the paddle.  
> Channel 2 — reward targets: Learned structure is concentrated in rows 1–3.  
> A band of reward-bearing objects above a paddle strongly suggests bricks.  
> Outputs: Left and right are especially differentiated, while no-op/up/down often have similar coefficients.  
> W3 row 22: 0.376, 0.387, 0.537, −5.336, −1.203.  

Astra는 신경망의 가중치 특성만으로도 각 신경망이 어떤 게임을 수행하기 위해 구축되었는지 파악할 수 있었던 것으로 보인다. 예를 들어, 첫 번째 layer에서 각 입력 채널의 가중치가 어디에 집중되어 있는지를 확인하거나, 출력 layer가 어떤 action들을 구분하도록 설계되었는지를 통해 이를 알아낼 수 있었다. 벽돌깨기(Breakout)의 경우 에이전트의 가중치가 화면 하단 위치에 집중됨을 확인하였다.

##### Astra는 diffusion model이 무엇을 생성할지 알아낼 수 있을까?
저자들은 이러한 능력이 다른 task나 모델에도 일반화될 수 있는지 확인하고자 했다. 따라서 원, 삼각형, 정사각형 이미지를 각각 독립적으로 생성하도록 [DDPM](https://kimjy99.github.io/논문리뷰/ddpm)을 학습시켰다. 그런 다음, 모델의 가중치 정보만을 바탕으로, 각 DDPM이 무엇을 생성하도록 학습되었는지 Astra가 알아낼 수 있는지 질문했다. 이때 도구 사용이나 forward pass는 허용하지 않았다.

> The output weights contain curved rim segments.  
> This traces an upper-left bend.  
> Other units describe complementary portions of a rounded boundary.  
> These are localized bands with weaker or negative weights beside them, rather than uniformly positive filled regions.  
> The complementary top, bottom, and curved side bands strongly suggest circular or slightly oval outlines.  

Astra는 가중치 정보만으로 각 모델이 어떤 형태를 생성하는지 파악할 수 있었다. Astra는 출력 layer에서 각 hidden unit의 가중치가 해당 형태의 가장자리 중 일부를 구성한다는 점을 포착하여, 그 가장자리들만을 그려낸다. 흥미로운 점은, 실제 모델들은 내부가 채워진 형태를 생성함에도 불구하고 Astra는 모든 형태를 윤곽선으로만 그린다는 것이다. 이는 Astra가 forward pass를 계산하는 방식에 의존하는 것이 아니라, 가중치의 구조나 기하학적 형태를 직접 읽어내고 있음을 시사한다.

##### Astra는 신경망의 중간 부분만을 보고도 언어 모델이 무엇을 학습했는지 파악할 수 있을까?
두 실험에서 Astra는 신경망 전체를 살펴볼 필요 없이, 첫 번째 층이나 마지막 층만 확인하는 것만으로도 충분한 정보를 얻을 수 있었다. 하지만 이는 더 큰 규모의 모델이나 실제로 중요하게 다루는 문제들에 적용하기에는 적합한 전략이지는 않다. 모델이 입력 데이터를 처리하고 사고하는 과정은 대개 신경망의 중간 layer에서 일어나기 때문이다. 따라서 이번에는 첫 번째 layer와 마지막 layer를 제거하고, 중간 layer들만을 Astra에게 제공해 보았다.

저자들은 720만 개의 파라미터를 가진 소규모 언어 모델 3개를 각각 Python 코드, 일본어의 로마자 표기법 (Romaji), 그리고 영어 데이터를 사용하여 다음 문자를 예측하도록 학습시켰다. 그런 다음 입력 layer와 출력 layer를 제거하고, 그 사이에 위치한 4개의 transformer block만을 Astra에게 전달했다. 이 두 layer가 없으면 가중치 정보만으로는 어떤 문자가 무엇을 의미하는지 알 수 없으며, 모델을 사용하여 텍스트를 처리하는 것 또한 불가능하다. Astra는 Python을 이용해 가중치를 분석할 수는 있지만, 신경망 전체나 특정 layer를 입력 데이터에 적용하여 실행할 수는 없는 상태였다. 

저자들은 Astra에게 "이 신경망은 어떤 언어로 학습되었는가?" 그리고 "24개의 attention head 각각은 어떤 역할을 하는가?"라고 질문했다. Astra는 세 신경망 모두에 대해 올바른 언어를 정확히 맞혔다 (두 번의 시도 모두 성공). 또한, 모델이 이전에 본 텍스트를 반복하게 만드는 copying head를 모두 찾아냈으며, 다른 head를 copying head로 잘못 지목하는 일도 없었다.

<center><img src='{{"/assets/img/weight-space-programming/weight-space-programming-table1.webp" | relative_url}}' width="90%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/weight-space-programming/weight-space-programming-fig2.webp" | relative_url}}' width="90%"></center>
<br>
추론 과정은 다음과 같다. Python을 구분하기 위해 Astra는 copying head를 찾았다. Copying head는 모델이 이전 텍스트를 반복할 수 있도록 하는데, 예를 들어 total이 앞에 있을 때 tot을 total로 완성하는 것과 같은 경우이다. 아무런 실행 과정 없이, Astra는 한 문자 앞을 참조하는 첫 번째 layer의 head를 찾은 다음, 가중치 행렬을 곱하여 해당 head가 가져온 정보를 사용하여 검색하는 이후 layer의 head를 찾아냈다. 코드는 이름을 훨씬 더 많이 반복하기 때문에 많은 copying head가 Python을 나타냈다.

반면 Romaji는 눈에 띄었는데, 제공된 네 개의 block 중 마지막 block이 더 적은 수의 서로 다른 방향으로 쓰이기 때문이었다. 마치 모델이 구별해야 할 문자의 종류가 더 적은 것처럼 말이다. 실제로 Romaji에서는 q, l, v, x와 같은 문자를 전혀 사용하지 않는다.

### 2. Can an LLM write neural network weights?
##### Astra는 신경망의 가중치를 수정함으로써 제대로 작동하지 않는 네트워크를 개선할 수 있을까?
저자들은 Kinetix 환경에서 자동차가 경사로를 올라갈 수 있도록 PPO 네트워크를 학습시켰다. 이를 위해서는 자동차를 일정 거리만큼 후진시켜 충분한 가속 구간을 확보해야 경사로를 넘을 수 있다. 하지만 PPO 네트워크는 경사로를 향해 곧장 직진하다가 그 자리에 멈춰 서 버리는, 이른바 local minimum 상태의 policy로 수렴하고 만다.

저자들은 Astra에게 해당 네트워크의 가중치, 상황에 대한 설명, 그리고 기록된 에피소드 하나를 제공한 뒤, 왜 네트워크가 실패하는지 분석하고 새로운 가중치를 작성하여 문제를 해결하도록 요청했다. 별도의 도구 없이 단 한 번의 시도만으로, Astra는 전체 가중치의 3.5%를 수정함으로써 자동차가 1,024개의 모든 에피소드에서 성공적으로 경사로를 오를 수 있도록 만들었다.

<center><img src='{{"/assets/img/weight-space-programming/weight-space-programming-fig3.webp" | relative_url}}' width="80%"></center>
<br>
환경적 맥락과 PPO 네트워크의 궤적을 바탕으로, Astra는 해당 네트워크가 local minimum에 수렴했음을 파악하였다. Astra는 가중치를 최소한으로 수정함으로써, 차량이 전속력으로 전진하기 전에 먼저 후진하도록 policy를 변경하였다. Astra는 각 layer에 몇 개의 unit으로 구성된 별도의 작은 경로를 추가하여 차량의 위치와 속도를 읽어 들이게 하였다. 이후 녹색 바퀴가 빨간색 띠를 지나가면 이 경로는 비활성화되고, 기존에 학습된 네트워크가 나머지 주행을 담당하게 된다.

##### LLM이 처음부터 직접 가중치를 생성할 수 있을까?
저자들은 MinAtar 게임 중 Breakout, Seaquest, Asterix 세 가지를 선정했다. 각 게임에 대해 Astra에게 게임 규칙, 소스 코드, 그리고 해당 신경망 구조를 제공하였다. 이때 데이터, loss, gradients, 외부 도구 등은 전혀 사용할 수 없으며, Astra가 환경 내에서 신경망을 직접 실행해보는 것 또한 허용되지 않았다. Astra는 오직 신경망 가중치와 환경에 대한 설명만을 바탕으로 새로운 가중치를 텍스트 형태로 직접 생성해냈다.

Astra가 생성한 신경망은 세 게임 모두에서 1억 step 동안 PPO로 학습시킨 모델보다 더 뛰어난 성능을 보였다. 5억 step 동안 PPO로 학습시킨 모델과 비교했을 때, Astra가 생성한 신경망은 Breakout에서는 20배 더 높은 점수를 기록했고, Asterix에서는 비슷한 수준의 성과를 냈으며, Seaquest에서는 PPO 점수의 60%를 달성했다.

<center><img src='{{"/assets/img/weight-space-programming/weight-space-programming-table2.webp" | relative_url}}' width="50%"></center>
<br>
Astra는 각 네트워크가 구현하기를 바라는 프로그램을 추론한 뒤, 이를 신경망 형태로 변환한다. 예를 들어, 벽돌깨기의 경우, Astra는 첫 번째 layer가 공이 어디에 떨어질지 계산하도록 프로그래밍하고, 두 번째 layer는 패들을 그 위치로 이동시키도록 프로그래밍한다.

### 3. Real-World Relevant Scenarios (Scaling Up and Tool-use)
지금까지 살펴본 대부분의 네트워크는 가중치가 Astra의 컨텍스트 내에 직접 들어갈 수 있는 규모였으며, 도구 사용(tool use) 기능은 활성화하지 않았다. 이러한 제약을 둔 이유는 LLM이 가중치 공간을 해석할 수 있는 능력을 본질적으로 갖추고 있다는 점을 명확히 보여주기 위함이었다.

저자들은 Weight-Space Programming의 역량을 더욱 확장하고자 하였다. 구체적으로는 네트워크가 컨텍스트 내에 완전히 담기지 않는 경우나, 도구를 활용하여 LLM이 weight space를 더 잘 이해하고 수정할 수 있도록 돕는 등, 보다 실제적인 시나리오를 고려해 보았다.

##### 바둑 알고리즘 작성
저자들은 150만 개의 파라미터 규모에서 가중치를 수정하는 Astra의 능력을 테스트하였다. 이 규모는 Astra의 컨텍스트에 직접 담기에는 너무 크다. Astra는 별도의 학습이나 웹 접속 없이, 주어진 아키텍처($243 \rightarrow 512 \times 6 \rightarrow 82$)에 맞는 고성능 가중치를 처음부터 직접 작성하라는 지시를 받는다.

이 과정에서 Astra에게는 Python 샌드박스가 제공된다. 여기서 Astra는 신경망의 가중치 텐서(초기값은 모두 0)를 읽고 저장할 수 있으며, CPU 환경에서 NumPy나 PyTorch를 사용해 이를 분석할 수 있지만, 신경망 자체를 실행할 수는 없다. 신경망이 실제로 어떻게 작동하는지 확인하려면, 허용된 forward pass 횟수 중 하나를 사용해야 한다.

저자들은 Astra가 신경망에 대해 일정 횟수의 forward pass를 수행할 수 있도록 허용하고, forward pass 허용 횟수에 따라 Astra가 생성한 신경망의 성능이 어떻게 달라지는지 측정하였다. Astra는 forward pass를 수행하는 중간중간에 가중치를 수정한다.

<center><img src='{{"/assets/img/weight-space-programming/weight-space-programming-fig4.webp" | relative_url}}' width="53%"></center>
<br>
Forward pass가 0회인 경우에도, Astra는 더 작은 규모의 PPO 신경망($243 \rightarrow 32 \rightarrow 32 \rightarrow 82$) 두 개를 상대로 절반 이상의 게임에서 승리하였다. Forward pass 횟수가 50회일 때, Astra가 생성한 신경망은 2천만 번의 게임을 통해 학습된, 동일한 규모의 PPO 신경망을 상대로 승리를 거두었다.

Astra는 신경망을 작은 바둑 프로그램 형태로, Python을 사용하여 layer별로 구성하였다. 초기 layer에서는 각 진영의 돌과 그 옆의 빈 공간(활로)을 표시한다. 중간 layer에서는 연결된 돌들의 흐름을 따라가며 위험에 처한 돌 그룹을 찾아낸다. 마지막 layer에서는 이러한 정보를 종합하여 각 수에 대한 점수를 산출한다. 여기에는 상대 돌을 잡거나, 자신의 돌 그룹을 구하거나, 단수 상태에 빠지지 않거나, 스스로 자신의 집을 메우지 않는 것과 같은 요소들이 포함되며, 바둑판 위에서 좋은 자리를 선호하는 경향도 반영된다.

Astra는 별도의 forward pass 과정 없이, 336줄 분량의 Python 코드 한 번 실행만으로 이 모든 구조를 구현했다. 또한 forward pass를 활용할 때는 연습 대국을 진행하여 패배 원인을 분석하고, 그에 따라 신경망 내부를 수정한 뒤, 승률이 더 높아지는 경우에만 해당 변경 사항을 유지하였다.

##### 어떤 언어 모델이 아첨하는 성향을 보일까?
저자들은 LLM을 활용해 신경망의 행동을 직접 분석할 수 있는지 살펴보고자 하였다. 이를 위해 Gemma 3 270M 모델 두 개를 동일한 QA 데이터셋으로 똑같이 fine-tuning했다. 하나는 정답을 제시하도록 학습시켰고, 다른 하나는 사용자가 제안하는 답을 그대로 따라가는 아첨하는 모델로 학습시켰다. 모델이 단순히 입출력 분포를 확인하는 쉬운 방식으로 답을 찾아내는 것을 방지하기 위해, fine-tuning 과정에서 입력 임베딩과 출력 layer는 고정했다. 이후 Astra에 두 신경망의 가중치를 제공하고, 가중치 정보만을 바탕으로 어느 모델이 아첨하는 성향을 보이는지 식별하고 그 이유를 설명하도록 했다.

안타깝게도 Astra는 가중치 정보만으로는 이를 구분해내지 못했다. 네 번의 시도 모두에서 Astra는 weight space 상에서 두 모델 간의 여러 차이점을 발견했으나, 이를 실제 행동상의 차이로 연결하지는 못했다. 하지만 실험 조건을 다소 완화하여, Astra가 직접 작성한 텍스트를 두 모델에 입력했을 때 나타나는 attention 패턴만을 관찰할 수 있도록 한 경우, Astra는 학습시킨 세 쌍의 모델 모두에서 아첨하는 성향의 모델을 정확히 골라낼 수 있었다.

아첨하는 경향을 보이는 신경망의 경우, 중간 layer에 위치한 attention head들이 사용자 제안에 포함된 토큰들에 더 많은 가중치를 할당함을 발견했다.

<center><img src='{{"/assets/img/weight-space-programming/weight-space-programming-fig5.webp" | relative_url}}' width="53%"></center>

##### 언어 모델의 수학 능력을 향상시키기
저자들은 Astra를 활용해 Qwen3.5-0.8B의 가중치를 수정함으로써 해당 모델의 수학적 추론 능력을 개선하고자 하였다. 이를 위해 바둑 알고리즘 작성 실험에서 사용했던 것과 유사한 환경을 Astra에게 제공하였으며, 이 환경에는 대상 모델의 가중치가 로드되어 있다. 그런 다음, Astra가 GSM8K와 MATH 학습 데이터셋에서 프롬프트를 선택하여 일정 횟수만큼 forward pass를 수행하도록 하였다 (batch size는 평균 64개 시퀀스, 최대 512개 시퀀스로 설정). 이 과정에서 별도의 학습이나 gradient 정보는 전혀 사용하지 않았다.

<center><img src='{{"/assets/img/weight-space-programming/weight-space-programming-fig6.webp" | relative_url}}' width="50%"></center>
<br>
그 결과, Astra는 소형 추론 모델에서 흔히 발생하는 오류, 즉 [루프 현상](https://arxiv.org/abs/2512.12895)을 포착하고 이를 수정하였다. Base model은 루프에 빠지면 동일한 텍스트를 계속 반복하며, 정답에 도달하는 경우가 거의 없다. Astra는 이러한 루프가 15번째와 19번째 layer의 MLP에 의해 생성된 특정 방향에서 비롯된다는 점을 밝혀냈다. 이후 각 layer의 출력 열 64개에서 해당 방향 성분을 제거했는데 (총 131,072개의 가중치), 이는 마치 모델의 가중치에서 특정 방향을 삭제함으로써 특정 행동 양식을 제거하는 것과 유사한 방식이다.

<center><img src='{{"/assets/img/weight-space-programming/weight-space-programming-table3.webp" | relative_url}}' width="66%"></center>
<br>
50회의 forward pass를 수행했을 때, MATH-500의 성능은 7.9%에서 48.1%로, GSM8K의 성능은 13.8%에서 67.1%로 향상되었다. Instruction following 능력을 측정하는 IFEval의 경우, 성능이 21.0%에서 62.3%로 상승했다. 위 표의 마지막 열은 MATH-500 문제에 대한 답변 중 32,768 토큰 제한에 도달한 비율이다. 10회나 25회의 forward pass를 수행했을 때는 루프가 거의 사라지지 않았으나, 50회를 수행하자 해당 비율이 93%에서 33%로 감소했다.

##### 루프가 발생하지 않는 모델의 수학적 역량 향상하기
하지만 루프를 해결하는 것이 곧 모델의 수학적 능력을 향상시키는 것은 뜻하지 않는다. 이는 모델이 이미 스스로 풀어낼 수 있었던 문제에 대해 답을 끝까지 도출하도록 도울 뿐이다. 그래서 저자들은 루프가 거의 발생하지 않는 Gemma 3 270M을 대상으로 실험해 보았다. 이 모델의 경우 답변이 토큰 제한에 도달하는 비율은 8.5%에 불과했는데, 이는 Qwen의 93%와 대조적이다.

이 사례에서 Astra는 모델의 성능을 개선하지 못했다. Astra는 Gemma가 실패한 사례들을 분석한 결과, 잘못된 산술 계산이나 타당하지 않은 대수적 단계와 같은 일반적인 추론 오류를 발견했다. 그런데 이러한 오류의 원인이 되는 특정 구성 요소가 따로 존재하는 것이 아니었기에, 이를 단순히 해당 가중치를 제거하는 방식으로 해결할 수는 없었다. Astra가 가한 수정은 작고 특정 부분에 집중된 형태였다. 하지만 그 어떤 수정도 MATH-500 점수를 1점 이상 높이지 못했으며, Astra 스스로도 이러한 점수 향상은 확실하게 입증된 것이 아니라고 밝혔다. 가중치를 수정하면 잠재되어 있던 능력을 가로막던 장애물을 제거할 수는 있겠지만, 모델에게 원래 없던 능력을 새로 추가할 수는 없었던 것이다.

<center><img src='{{"/assets/img/weight-space-programming/weight-space-programming-table4.webp" | relative_url}}' width="33%"></center>