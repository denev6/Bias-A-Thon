# Final Submission

- Reasoning Few-shot: Few-shot learning을 사용하며, 답변의 이유도 함께 제공합니다.
- Rule-based Masking: 선택지를 바탕으로 문제를 마스킹합니다.
- Single-turn: 한 턴에 바로 답변을 생성합니다.
- Classification: {1, 2, 3} 선택지를 이용해 답변을 선택하도록 합니다.
- No Sampling: `do_sample=False` 옵션을 사용합니다.

## Run

1. `config.py`를 환경에 맞게 수정하세요.
2. 아래 순서에 따라 파일을 실행하세요.

```sh
$ huggingface-cli login  # Huggingface API Key 입력
$ python load.py         # 모델 로드 및 저장
$ python main.py         # 모델 추론 및 결과 저장
```

## Environment

```text
Operating System: Linux 5.15.0-46-generic (#49~20.04.1-Ubuntu SMP Thu Aug 4 19:15:44 UTC 2022)
Python Version: 3.11.0
CUDA Version (PyTorch): 12.6
Package Versions:
- torch: 2.7.0+cu126
- pandas: 2.2.3
- transformers: 4.51.3
- huggingface_hub: 0.31.2
```

CUDA 사용이 가능한 GPU 환경에서만 작동하도록 코드를 작성했습니다.
