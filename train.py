# Some parts of the code are taken from https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb and HuggingFace/open-r1 Implementation

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from utils import *
from reward_funcs import *
from load_dataset import get_gsm8k_questions
import torch
import os
# os.environ["WORLD_SIZE"] = "1"  # Ensure only one process is used

# Load and prep dataset
train_dataset = get_gsm8k_questions(split="train")
test_dataset = get_gsm8k_questions(split="test")


training_args = GRPOConfig(
    output_dir="outputs/Llama-1B-base-GRPO",
    run_name="Llama-1B-base-GRPO-gsm8k",
    learning_rate=5e-7,
    adam_beta1 = 0.9,
    adam_beta2 = 0.95,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=786,
    max_grad_norm=0.01,
    # report_to="wandb",
    log_on_each_node=False,
    use_vllm=True,
    vllm_device="cuda:0",
    vllm_gpu_memory_utilization=0.30,
    bf16=True,
    # torch_empty_cache_steps=1,
    gradient_checkpointing=True
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
else:
    output_dir="outputs/Qwen-1.5B-GRPO"
    run_name="Qwen-1.5B-GRPO-gsm8k"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    # device_map="cuda:0"  # Use only one GPU
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
# )

# model = get_peft_model(model, peft_config)
# model = get_peft_model(model, peft_config)
# print(model.print_trainable_parameters())

# tokenizer = AutoTokenizer.from_pretrained(model_name + '-Instruct')
# tokenizer.pad_token = tokenizer.eos_token
# model.to("cuda:0")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=test_dataset,
    # peft_config=peft_config
)
trainer.train()