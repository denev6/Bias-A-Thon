# Import
import os
import re
import ast
import gc
import time
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="./baseline")
parser.add_argument("--model_dir", type=str, default="./Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--checkpoint_interval", type=int, default=100)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--use_compile", action="store_true")
parser.add_argument("--input_data", type=str, default="./test.csv")
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

# Data Load
print("Loading dataset...")
data = pd.read_csv(args.input_data, encoding="utf-8-sig")


# Prompt Function
def generate_first_prompt(row):
    context = row["context"]
    question = row["question"]
    choices = ast.literal_eval(row["choices"])

    context = context.replace(choices[0], "A").replace(choices[1], "B")

    return (
        f"<|start_header_id|>user<|end_header_id|>문맥: {context.strip()}\n"
        f"질문: {question.strip()}\n"
        "선택지:\n"
        "- 1. A\n"
        "- 2. B\n"
        "- 3. 알 수 없음\n"
        "답:<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>"
    )


# Preprocess Prompts
print("Preprocessing prompts...")
with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    prompt_futures = {
        executor.submit(generate_first_prompt, row): i for i, row in data.iterrows()
    }

    prompts = [None] * len(data)
    for future in tqdm(
        as_completed(prompt_futures), total=len(data), desc="Creating prompts"
    ):
        idx = prompt_futures[future]
        prompts[idx] = future.result()

data["prompt"] = prompts

# Model Load
print("Loading model and tokenizer...")
model_kwargs = {
    "device_map": "auto",  # 자동 분산
    "trust_remote_code": True,
    "local_files_only": True,
    "torch_dtype": torch.float16,
}

tokenizer = AutoTokenizer.from_pretrained(
    args.model_dir,
    local_files_only=True,
    trust_remote_code=True,
)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_dir, low_cpu_mem_usage=True, **model_kwargs  # GPU 메모리 절약 로딩
)

if args.use_compile and hasattr(torch, "compile"):
    try:
        print("Compiling model...")
        model = torch.compile(model)
    except Exception as e:
        print(f"Model compile failed: {e}")


# Utilities
def extract_last_choice(raw_answer, choices):
    first_digit = next((char for char in raw_answer if char.isdigit()), None)
    if first_digit and first_digit.isdigit():
        idx = int(first_digit)
        if 1 <= idx <= 3:
            return choices[idx - 1]
    return raw_answer.strip()


@torch.no_grad()
def batch_tokenize(prompts):
    return tokenizer(
        prompts, padding=True, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)


@torch.no_grad()
def process_batch(prompts):
    inputs = batch_tokenize(prompts)
    outputs = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,  # safe decoding
        num_beams=1,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Inference
def predict_with_dynamic_batching(df):
    all_results = []
    total = len(df)
    current_batch_size = args.batch_size
    max_batch_size = 16
    start_idx = 0

    with tqdm(total=total, desc="Processing") as pbar:
        while start_idx < total:
            end_idx = min(start_idx + current_batch_size, total)
            batch = df.iloc[start_idx:end_idx]
            batch_prompts = batch["prompt"].tolist()

            try:
                start_time = time.time()
                batch_raw_outputs = process_batch(batch_prompts)
                elapsed = time.time() - start_time
                time_per_sample = elapsed / len(batch_prompts)

                for i, raw_output in enumerate(batch_raw_outputs):
                    row = batch.iloc[i]
                    choices = ast.literal_eval(row["choices"])
                    final_answer = extract_last_choice(raw_output, choices)
                    all_results.append(
                        {
                            "ID": row["ID"],
                            "raw_input": batch_prompts[i],
                            "raw_output": raw_output,
                            "answer": final_answer,
                        }
                    )

                pbar.update(len(batch_prompts))
                pbar.set_postfix(
                    batch_size=current_batch_size, tps=f"{time_per_sample:.2f}s"
                )

                if time_per_sample < 0.5 and current_batch_size < max_batch_size:
                    current_batch_size = min(current_batch_size + 2, 32)
                elif time_per_sample > 2.0 and current_batch_size > 2:
                    current_batch_size = max(current_batch_size - 2, 2)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nOOM at batch {start_idx}. Reducing batch size.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    current_batch_size = max(current_batch_size // 2, 1)
                    continue
                else:
                    raise e

            start_idx = end_idx

            if start_idx % args.checkpoint_interval == 0:
                print(f"Checkpoint {start_idx}/{total}")
                pd.DataFrame(all_results).to_csv(
                    os.path.join(args.save_dir, f"checkpoint_{start_idx}.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )

            if start_idx % (args.checkpoint_interval // 2) == 0:
                torch.cuda.empty_cache()
                gc.collect()

    return pd.DataFrame(all_results)


# Start
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("🏁 Starting inference...")
start_time = time.time()

try:
    result_df = predict_with_dynamic_batching(data)
    result_df.to_csv(
        os.path.join(args.save_dir, "submission.csv"), index=False, encoding="utf-8-sig"
    )
    total_time = time.time() - start_time
    print(
        f"Inference done! Time: {total_time:.2f}s, Speed: {len(data)/total_time:.2f} samples/s"
    )

except Exception as e:
    print(f"Inference failed: {e}")
    import traceback

    traceback.print_exc()

# Cleanup
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()
