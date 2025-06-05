# Bias-A-Thon 2025: Track 2

⚠️ `data/Test_Data_Answer_{100, 200}.csv`도 대회 측 데이터이기 때문에 Repo 공개 전 삭제할 예정입니다. (reinforcement 내 데이터도 삭제)

이 저장소는 [Bias-A-Thon 2025: Track 2](https://dacon.io/competitions/official/236487/overview/description) 대회에서 **최종 3위를 기록한 솔루션**입니다.

## 📌 대회 소개

`Bias-A-Thon`은 대규모 언어 모델(LLM)의 **사회적·문화적 편향 문제를 해결**하는 것을 목표로 하는 대회입니다. `Track 2`는 LLaMA 3.1 8B-Instruct 모델을 활용해 다음 과제를 해결합니다:

- 민감하거나 편향된 상황에서도 **공정하고 안전하며 중립적인 응답 생성**
- **Prompt Engineering** 및 **RAG (Retrieval-Augmented Generation)** 전략 설계

## 🥉 아이디어

저희 팀은 다음과 같은 아이디어를 통해 최종 3위를 달성했습니다:

- **Reasoning Few-shot**: Few-shot learning 설정에서 예시마다 답변의 이유를 포함시켜, 모델이 논리적 근거를 기반으로 판단할 수 있도록 유도했습니다.
- **Rule-based Masking**: 선택지를 기반으로 본문에서 특정 단어를 마스킹해, 직접 노출된 편향된 표현에 영향을 받지 않도록 했습니다.
- **Single-turn**: 멀티턴 대화 없이 한 번의 프롬프트로 답변을 생성했습니다.
- **Classification**: 최종 응답은 {1, 2, 3} 중 하나의 선택지를 고르도록 유도하여 채점이 용이하도록 했습니다.
- **Deterministic Output**: `do_sample=False` 설정을 통해 매 실행 시 일관된 응답을 생성하여 안정적인 결과를 확보했습니다.

### 제출 결과

|폴더|Public 점수|
|:---:|:---:|
|[final_submission](final_submission)|0.9227|
|[dynamic_temperature](dynamic_temperature)|0.9149|
|[reasoning_fewshot](reasoning_fewshot)|0.9116|

## 관련 링크

- 🤖 [Bias-A-Thon : Bias 대응 챌린지 <Track 2>](https://dacon.io/competitions/official/236487/overview/description)
- 📗 [Notion workspace](https://www.notion.so/1ea94c27e48280db9584f4ebf7f83aa7?pvs=4)
- 📦 [Google Drive](https://drive.google.com/drive/folders/18vzXbeDobmMidoomdQO16w3Wg_n8vyB9?usp=sharing): 학교 구글 계정(@g.skku.edu)으로 접속하면 볼 수 있습니다.

## 참고 자료

- 데이터 설명: [data](data)
- 풀고자 하는 문제: [examples/problem](examples/problem.ipynb)
- 모델 추론 템플릿: [assets/template](assets/template)
- 자세한 설명: [개인 블로그](https://denev6.github.io/projects/2025/05/24/dacon-bias.html)
