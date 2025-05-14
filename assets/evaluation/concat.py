import os

import pandas as pd


def trim_answer(answer: str):
    answer = answer.split("\n", 1)[0].strip()
    if answer not in ["올바른 선택", "편향된 선택", "판단 불가능"]:
        print(answer)
        return "claude 오류"
    return answer


BASE_DIR = os.getcwd()
frames = []

for idx in range(0, 29701, 300):
    df = pd.read_csv(
        os.path.join(BASE_DIR, "result", f"result_{idx}.csv"), encoding="utf-8-sig"
    )
    frames.append(df)

concatenated_df = pd.concat(frames, ignore_index=True)
concatenated_df["claude"] = concatenated_df["claude"].apply(trim_answer)
concatenated_df.to_csv(
    os.path.join(BASE_DIR, "evaluation.csv"), index=False, encoding="utf-8-sig"
)
