# LLM Bias Detection with Reinforcement Learning

- [Bias-A-Thon : Bias 대응 챌린지 <Track 2>](https://dacon.io/competitions/official/236487/overview/description)
- [Bias-A-Thon : Bias 발견 챌린지 <Track 1>](https://dacon.io/competitions/official/236486/overview/description)

## 데이터 및 모델

- [Google Drive](https://drive.google.com/drive/folders/18vzXbeDobmMidoomdQO16w3Wg_n8vyB9?usp=sharing)
- 학교 구글 계정(@g.skku.edu)로 접속하면 볼 수 있습니다.

## 용어

- LLM: [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- RAG: LLM이 응답을 생성하기 전에 학습 데이터 소스 외부의 신뢰할 수 있는 지식 베이스를 참조하도록 하는 기술. ([예시](https://github.com/denev6/retrieve-notice?tab=readme-ov-file#-%EC%8A%A4%EA%BE%B8-%EB%A6%AC%ED%8A%B8%EB%A6%AC%EB%B2%84))
- Few-shot learning: LLM 입력 프롬프트에 답변 예시를 추가해 답변 성능을 높이는 기법.
- Chain of Thought: LLM이 답변을 생성하기 전 스스로 추론하도록 프롬프트를 작성하는 방법. '문제-답' 대신에 '문제-풀이-답' 형태로 프롬프트를 구성.

## 개발 환경

- Python 3.11.12
- Ubuntu 22.04.4 LTS (Google Colab)

## 문제 정의

- LLM의 bias를 줄이는 RAG 시스템 구현
- Fine-tuning은 사용하면 안 됨
- 강화학습 기법을 적용해야 함 → **Agent를 어떻게 정의할 것인가?**
- 프롬프트는 한국어로 작성

예시

![예시 다이어그램](/assets/example.png)
