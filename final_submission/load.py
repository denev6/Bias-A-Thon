import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import *

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

quat_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=MODEL_DEVICE_MAP,
    quantization_config=quat_config,
    torch_dtype=torch.float16,
)

tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)
