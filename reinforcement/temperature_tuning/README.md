# Temperature tuning with Reinforcement learning

![PPO 학습 과정](./ppo-overview.png)

## 데이터

- `train.csv`: GPT로 생성한 데이터셋 400개 (학습용)
- `Test_Data_Answer_200.csv`: 대회 측 데이터 200개 (평가용)

## 학습 결과

```text
Evaluating model for 2 episodes...

Episode 1:
  Episode reward: 107.000
  Average temperature: 0.001

Episode 2:
  Episode reward: 105.000
  Average temperature: 0.001

==============================
EVALUATION RESULTS
==============================
Average episode reward: 106.000 ± 1.000
Average temperature: 0.001 ± 0.000
```
