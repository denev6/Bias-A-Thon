import os
import gc
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def set_cuda(random_seed):
    assert torch.cuda.is_available(), "CUDA를 사용할 수 없습니다!"
    device = "cuda"

    # CUDA 디버깅
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # CUDA 최적화
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "matmul") and hasattr(
        torch.backends.cuda.matmul, "allow_tf32"
    ):
        torch.backends.cuda.matmul.allow_tf32 = True

    # 랜덤 시드 고정
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return device


def collect_garbage():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


class Model:
    def __init__(
        self,
        path,
        device,
        max_length,
        do_sample,
        temperature=0.6,
        top_k=30,
        top_p=0.9,
        repetition_penalty=1.0,
        skip_special_tokens=True,
        device_map="auto",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        quat_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=device_map,
            quantization_config=quat_config,
            torch_dtype=torch.float16,
        )
        self.max_length = max_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.skip_special_tokens = skip_special_tokens
        self.device = device

    @torch.no_grad()
    def tokenize_batch(self, batch_prompts):
        return self.tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

    @torch.no_grad()
    def process_batch(self, batch_tokens, max_new_tokens):
        answer_tokens = self.model.generate(
            input_ids=batch_tokens["input_ids"],
            attention_mask=batch_tokens["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        decoded_answers = self.tokenizer.batch_decode(
            answer_tokens, skip_special_tokens=self.skip_special_tokens
        )
        return decoded_answers
