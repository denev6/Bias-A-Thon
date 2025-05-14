import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_last_choice(raw_answer):
    first_digit = next((char for char in raw_answer if char.isdigit()), None)
    if first_digit is None:
        return "알 수 없음"

    if first_digit.isdigit():
        last_choice_idx = int(first_digit)
        if 1 <= last_choice_idx <= 2:
            return f"{{선택{last_choice_idx}}}"

    return "알 수 없음"


def trim_raw_input(row):
    raw_input = row["raw_input"]
    raw_output = row["raw_output"]
    _, question = raw_input.rsplit("user", 1)
    question, _ = question.rsplit("선택지:", 1)
    question = question.strip("\n").strip()
    output = get_last_choice(raw_output)
    return f"{question}\n답: {output}"


def preprocess(data_frame, function, num_workers):
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(function, row) for _, row in data_frame.iterrows()]
        for future in as_completed(futures):
            results.append(future.result())
    return results


df = pd.read_csv("submission_reasoning.csv")
df["summary"] = ""

chunk_size = 300
start_idx = 0
total_size = len(df)

while start_idx < total_size:
    end_idx = min(start_idx + chunk_size, total_size)
    df_chunk = df.iloc[start_idx:end_idx].copy()
    preprocessed = preprocess(df_chunk, trim_raw_input, 2)
    df_chunk.loc[:, "summary"] = preprocessed
    df_chunk[["ID", "summary"]].to_csv(
        f"chuck_{start_idx}.csv", index=False, encoding="utf-8-sig"
    )
    print(f"saved {start_idx}")

    start_idx += chunk_size

print("=" * 20)
print(f"saved all")
