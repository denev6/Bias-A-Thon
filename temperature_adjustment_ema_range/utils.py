import os
import warnings

import pandas as pd

from config import BASE_DIR


def ignore_warnings():
    warnings.filterwarnings("ignore")


def join_path(*args):
    return os.path.join(BASE_DIR, *args)


def save_data(data_frame, cols, path, index=False, encoding="utf-8-sig"):
    data_frame[cols].to_csv(
        path,
        index=index,
        encoding=encoding,
    )


def load_data(
    path,
    checkpoint_dir,
    cols,
    prefix="submission",
    last_checkpoint=0,
    encoding="utf-8-sig",
):
    """데이터 가져오기

    Returns:
      - 원본 데이터
      - check_point 데이터 (없으면 생성)
      - 체크포인트
    """
    df_original = pd.read_csv(join_path(path), encoding=encoding)

    # Check point 확인
    os.makedirs(join_path(checkpoint_dir), exist_ok=True)
    check_point_path = join_path(checkpoint_dir, f"{prefix}_{last_checkpoint}.csv")
    check_point = last_checkpoint

    if os.path.exists(check_point_path):
        df_check_point = pd.read_csv(check_point_path)
    else:
        # Check point가 없을 때 초기화
        df_check_point = df_original.copy()
        check_point = 0
        for col in cols:
            if col not in df_check_point.columns:
                df_check_point[col] = ""
            df_check_point[col] = df_check_point[col].astype("string")

    df_check_point = df_check_point[cols]
    return df_original, df_check_point, check_point
