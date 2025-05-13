"""5-shot learning with Human-persona"""

import ast
import time

from config import *
from model import Model, set_cuda
from prompt import preprocess, generate_masked_prompt, split_answer, extract_last_choice
from utils import ignore_warnings, load_data, save_data, join_path


# 수행할 작업
def pipeline(model, batch_prompts) -> list[str]:
    question_tokens = model.tokenize_batch(batch_prompts)
    answer_tokens = model.process_batch(question_tokens)
    return answer_tokens


print("🔥설정 준비 중...")
if IGNORE_WARNING:
    ignore_warnings()

# CUDA 설정
device = set_cuda()

# Llama3
print("🔥모델 불러오는 중...")
our_llm = Model(
    join_path(MODEL_DIR),
    device=device,
    max_length=TOKENIZER_MAX_LENGTH,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=DO_SAMPLE,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    skip_special_tokens=SKIP_SPECIAL_TOKENS,
    device_map=MODEL_DEVICE_MAP,
)

# 사용할 데이터 및 체크포인트
print("🔥데이터 준비 중...")
df_original, df_check_point, start_idx = load_data(
    path=join_path(INPUT_DATA),
    checkpoint_dir=join_path(CHECKPOINT_DIR),
    last_checkpoint=LAST_CHECK_POINT,
)
total_data_size = len(df_check_point)

# 프롬프트 준비
prompts = preprocess(
    df_original,
    function=generate_masked_prompt,
    num_workers=NUM_WORKERS,
)

# 추론 및 저장
print("🔥모델 추론을 시작합니다!")
start_time = time.time()
while start_idx < total_data_size:
    end_idx = min(start_idx + BATCH_SIZE, total_data_size)
    batch_prompts = prompts[start_idx:end_idx]
    batch_answers = pipeline(our_llm, batch_prompts)

    for idx, answer in enumerate(batch_answers):
        idx = idx + start_idx
        prompt, raw_answer = split_answer(answer)
        df_check_point.at[idx, "raw_input"] = prompt
        df_check_point.at[idx, "raw_output"] = raw_answer
        choices = ast.literal_eval(df_original.at[idx, "choices"])
        df_check_point.at[idx, "answer"] = extract_last_choice(raw_answer, choices)

        if idx % CHECK_POINT_STEP == 0:
            # Check point에서 답변을 파일로 저장
            end_time = time.time()
            save_data(
                df_check_point,
                cols=["ID", "raw_input", "raw_output", "answer"],
                path=join_path(CHECKPOINT_DIR, f"submission_{str(idx)}.csv"),
            )
            print(
                f"✅{idx}/{total_data_size} 저장. ({(end_time - start_time) / 60:.1f}분)"
            )
            start_time = time.time()

    start_idx = end_idx


# 최종 파일 저장
save_data(
    df_check_point,
    cols=["ID", "raw_input", "raw_output", "answer"],
    path=join_path(OUTPUT_FILE),
)
print("\n🫠기록이 완료되었습니다.")
