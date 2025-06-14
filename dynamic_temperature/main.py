"""Reasoning 5-shot learning with dynamic temperature"""

import ast
import time

from config import *
from model import Model, set_cuda, collect_garbage
from prompt import (
    preprocess,
    generate_full_prompt,
    split_answer,
    extract_last_choice,
)
from utils import ignore_warnings, load_data, save_data, join_path


# 수행할 작업
def pipeline(model, batch_prompts, max_new_tokens) -> list[str]:
    question_tokens = model.tokenize_batch(batch_prompts)
    answer_tokens = model.process_batch(question_tokens, max_new_tokens)
    return answer_tokens


def get_reward(output: str, min_len: int = 60, max_len: int = 110) -> int:
    """답변 길이에 따라 보상 계산"""
    length = len(output)
    if min_len < length < max_len:
        return +1
    return -1  # 너무 짧거나 너무 길면 패널티


def clip_temperature(temp, min_temp=0.1, max_temp=0.6):
    return min(max_temp, max(min_temp, temp))


print("🔥설정 준비 중...")
if IGNORE_WARNING:
    ignore_warnings()

# CUDA 설정
device = set_cuda(RANDOM_SEED)

# Llama3
print("🔥모델 불러오는 중...")
our_llm = Model(
    join_path(MODEL_DIR),
    device=device,
    max_length=TOKENIZER_MAX_LENGTH,
    do_sample=DO_SAMPLE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    repetition_penalty=REPETITION_PENALTY,
    skip_special_tokens=SKIP_SPECIAL_TOKENS,
    device_map=MODEL_DEVICE_MAP,
)


file_prefix = "submit"

df_original, df_check_point, start_idx = load_data(
    path=join_path(INPUT_CSV),
    cols=["ID", "raw_input", "raw_output", "answer"],
    checkpoint_dir=join_path(CHECKPOINT_DIR),
    last_checkpoint=LAST_INFERENCE_CHECK_POINT,
    prefix=file_prefix,
)
total_data_size = len(df_check_point)

# 마스킹된 프롬프트 준비
prompts = preprocess(
    df_original,
    function=generate_full_prompt,
    num_workers=NUM_WORKERS,
)

# 추론 및 저장
print("🔥모델 추론을 시작합니다!")
collect_garbage()

start_time = time.time()

current_temperature = TEMPERATURE
scale_factor = 0.05

while start_idx < total_data_size:
    end_idx = min(start_idx + BATCH_SIZE, total_data_size)
    batch_prompts = prompts[start_idx:end_idx]
    batch_answers = pipeline(our_llm, batch_prompts, max_new_tokens=MAX_NEW_TOKENS)

    for idx, answer in enumerate(batch_answers):
        idx = idx + start_idx

        # 모델의 최종 응답에서 전체 답변 텍스트 추출
        prompt, raw_answer = split_answer(answer)
        # 길이에 따른 보상 계산
        reward = get_reward(raw_answer)
        choices = ast.literal_eval(df_original.at[idx, "choices"])
        extracted_answer = extract_last_choice(raw_answer, choices)

        # 결과 기록
        df_check_point.at[idx, "raw_input"] = prompt
        df_check_point.at[idx, "raw_output"] = raw_answer
        df_check_point.at[idx, "answer"] = extracted_answer
        df_check_point.at[idx, "reward"] = reward

        # [NEW] 온도 조정 (보수적으로)
        current_temperature = current_temperature * (1 + reward * scale_factor)
        current_temperature = clip_temperature(current_temperature)
        our_llm.temperature = current_temperature

        if idx % CHECK_POINT_STEP == 0:
            # Check point에서 답변을 파일로 저장
            end_time = time.time()
            save_data(
                df_check_point,
                cols=["ID", "raw_input", "raw_output", "answer"],
                path=join_path(CHECKPOINT_DIR, f"{file_prefix}_{str(idx)}.csv"),
            )
            print(f"✅[{idx}/{total_data_size}] {(end_time - start_time) / 60:.1f}분")
            start_time = time.time()

    start_idx = end_idx


# 최종 파일 저장
save_data(
    df_check_point,
    cols=["ID", "raw_input", "raw_output", "answer"],
    path=join_path(FINAL_CSV),
)
print("\n🫠기록이 완료되었습니다.")
