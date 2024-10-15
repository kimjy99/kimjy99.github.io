---
title: "[AI소식] LLM Colosseum"
last_modified_at: 2024-04-10
categories:
  - AI소식
tags:
  - NLP
  - LLM
  - AI
excerpt: "LLM Colosseum"
use_math: true
classes: wide
---

> [[Github](https://github.com/OpenGenerativeAI/llm-colosseum)] [[Hugging Face](https://huggingface.co/spaces/junior-labs/llm-colosseum)] [[해외기사](https://www.tomshardware.com/tech-industry/artificial-intelligence/fourteen-llms-fight-it-out-in-street-fighter-iii-ai-showdown-finds-out-which-models-make-the-best-street-fighters)]  

3월 23일, Mistral Hackathon 2024에서 스트리트 파이터 3를 기반으로 한 LLM 벤치마크인 **LLM Colosseum**이 제안되었다. 각 플레이어는 LLM에 의해 제어되며 화면에 대한 텍스트 설명이 LLM에게 입력된다. LLM은 이전 동작, 상대의 동작, 파워 및 체력바를 고려하여 수행할 다음 동작을 결정한다. 

실험은 Mistral AI의 LLM 모델 3개와 OpenAI의 GPT 모델 5개로 진행되었다. 각 경기 결과에 따라 ELO를 반영하였으며, 총 342번의 매치 후 Open AI의 GPT 3.5 Turbo가 우승을 차지하였다. 

### 경기 결과
ELO 랭킹은 아래 표와 같다. 작은 모델이 지연시간과 속도 측면에서 유리하였다. 

<center><img src='{{"/assets/img/llm-colosseum/elo.PNG" | relative_url}}' width="35%"></center>
<br>
모델들 간의 승률을 아래와 같다. 

<center><img src='{{"/assets/img/llm-colosseum/win_rate_matrix.png" | relative_url}}' width="70%"></center>

### 입력 프롬프트
LLM에 입력되는 프롬프트는 다음과 같다. 

> <pre>You are the best and most aggressive Street Fighter III 3rd strike player in the world.  
> Your character is {character}. Your goal is to beat the other opponent. You respond with a bullet point list of moves.  
> {position_prompt}  
> {power_prompt}  
> {last_action_prompt}  
> Your current score is {reward}. {score_prompt}  
> To increase your score, move toward the opponent and attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.  
> The moves you can use are:  
> {move_list}  
> ----  
> Reply with a bullet point list of moves. The format should be: `- &lt;name of the move&gt;` separated by a new line.  
> Example if the opponent is close:  
> - Move closer  
> - Medium Punch  
>  
> Example if the opponent is far:  
> - Fireball  
> - Move closer  

`position_prompt`는 적과 멀리 있는지 가까이 있는지를 알려주며, 멀리 있다면 적의 위치와 함께 적에게 이동하라는 명령이 주어지고, 가까이 있다면 공격하라는 명령이 주어진다. 

> <pre>You are very far from the opponent. Move closer to the opponent. Your opponent is on the left.  
> You are very far from the opponent. Move closer to the opponent. Your opponent is on the right.  
> You are close to the opponent. You should attack him.  

`power_prompt`는 사용 가능한 슈퍼콤보를 알려준다. 

> <pre>You can now use a powerfull move. The names of the powerful moves are: Megafireball, Super attack 2.  
> You can now only use very powerfull moves. The names of the very powerful moves are: Super attack 3, Super attack 4  

`last_action_prompt`는 플레이어와 적의 마지막 동작을 알려준다. 

> <pre>Your last action was {str_act_own}. The opponent's last action was {str_act_opp}.

`score_prompt`는 reward에 따라 이기고 있는지 지고 있는지를 알려준다. 

> <pre>You are winning. Keep attacking the opponent.  
> You are losing. Continue to attack the opponent but don't get hit.  

`move_list`는 사용 가능한 동작들을 알려준다. 

### 별도 테스트
추가로, Amazon의 Banjo Obayomi가 진행한 별도의 테스트에서는 14개의 LLM이 314번의 매치를 진행했고 Anthropic의 Claude 3 Haiku가 우승하였다고 한다 (ELO 1613). 

Hallucination이나 AI의 안전 장치들이 격투게임 성능을 저하시키는 경우도 있었다고 한다. 