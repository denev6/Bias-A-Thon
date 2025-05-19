## 주요 설정
BASE_DIR = "./"  # 작업 경로를 설정해 주세요
INPUT_CSV = "test.csv"  # 입력 csv
FINAL_CSV = "final.csv"  # 출력 csv
MODEL_DIR = "llama3"  # 모델 디렉토리
MODEL_DEVICE_MAP = "auto"  # GPU 상황에 맞게 변경하세요
BATCH_SIZE = 32  # CUDA-Out-of-Memory가 발생하면 줄여주세요
CHECKPOINT_DIR = "checkpoint"
LAST_INFERENCE_CHECK_POINT = 0  # (int) 체크포인트 (없으면 처음부터 탐색)

## 세부 설정
DEFAULT_CHOICE = "알 수 없음"
NUM_WORKERS = 2
IGNORE_WARNING = True
SKIP_SPECIAL_TOKENS = True
DO_SAMPLE = False
MAX_NEW_TOKENS = 64
TOKENIZER_MAX_LENGTH = 2048
CHECK_POINT_STEP = 500
RANDOM_SEED = 42
