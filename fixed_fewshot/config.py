## 주요 설정
BASE_DIR = "/root/workspace/bias"  # 작업 경로를 설정해 주세요
INPUT_CSV = "test.csv"  # 입력 csv
MASKED_CSV = "masked.csv"  # 중간 마스킹을 저장할 csv
FINAL_CSV = "submission_rl_final1.csv"  # 출력 csv
MODEL_DIR = "llama3"  # 모델 디렉토리
CHECKPOINT_DIR = "checkpoint"
MODEL_DEVICE_MAP = "auto"  # GPU 상황에 맞게 변경하세요
# (int) 이전에 저장한 체크포인트 (없으면 자동으로 처음부터 탐색)
LAST_INFERENCE_CHECK_POINT = 0
BATCH_SIZE = 32  # CUDA-Out-of-Memory가 발생하면 2배수로 줄여가며 실행해 보세요

## 세부 설정
DEFAULT_CHOICE = "알 수 없음"
NUM_WORKERS = 2
IGNORE_WARNING = True
SKIP_SPECIAL_TOKENS = True
DO_SAMPLE = False
TEMPERATURE = None
TOP_P = None
TOP_K = None
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 64
TOKENIZER_MAX_LENGTH = 2048
CHECK_POINT_STEP = 500
RANDOM_SEED = 42
