# 프롬프트 기법

## Self-debiasing

Few-shot, CoT 등 방법보다 좋은 결과를 보였다\[1\]. 참고로 여기서 CoT는 프롬프트에 "Think step by step"을 추가한 경우를 말한다.

**알고리즘**:

```python
is_biased =  bias_determination()
while is_biased:
    bias_analysis()
    cognitive_debiasing()
    is_biased =  bias_determination()

generate_answer()
```

- 편향이 없으면 바로 LLM 응답을 생성한다.
- 편향이 있다고 판단하면 `Bias Analysis`와 `Congitive Debiasing`을 거친 후 다시 편향을 검사한다.

**사용한 프롬프트**:

```text
[Bias determination]
Please first break prompt into sentence by sentence, 
and then determine whether may contain cognitive biases that affect normal decision.

[Bias analysis]
The following is a task prompt may contain cognitive biases. 
Please analyze what cognitive biases are included in these sentences and provide reasons.

[Cognitive debiasing]
The following task prompt may contain cognitive biases. 
Rewrite the prompt according to the bias judgment such that a human is not biased, 
while retaining the normal task.
```

## Human persona

LLM에게 사람의 페르소나를 부여하고, 천천히 생각하도록 유도하면 편향이 줄어드는 결과를 보였다\[2\].

**사용한 프롬프트**:

```text
[HP + System 2]
Adopt the identity of a person who answers questions slowly and thoughtfully.
Their answers are effortful and reliable.
Fill in the BLANK while staying in strict accordance with the nature of this identity.
Given the context below, ...
```

## 참고문헌

1. [Cognitive Debiasing Large Language Models for Decision-Making](https://arxiv.org/pdf/2504.04141), 2025
1. [Prompting Techniques for Reducing Social Bias in LLMs through System 1 and System 2 Cognitive Processes](https://arxiv.org/pdf/2404.17218v1), 2024
1. [Prompting Fairness: Integrating Causality to Debias Large Language Models](https://arxiv.org/pdf/2403.08743), 2025
1. [Rethinking Prompt-based Debiasing in Large Language Models, 2025](https://arxiv.org/pdf/2503.09219)
1. [MindScope: Exploring cognitive biases in large language models through Multi-Agent Systems](https://arxiv.org/pdf/2410.04452v1), 2024
1. [Thinking Fair and Slow: On the Efficacy of Structured Prompts for Debiasing Language Models](https://arxiv.org/pdf/2405.10431), 2024
