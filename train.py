# Some parts of the code are taken from https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb and HuggingFace/open-r1 Implementation

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from utils import *
from reward_funcs import *
from load_dataset import get_gsm8k_questions
import torch

# Load and prep dataset
dataset = get_gsm8k_questions()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

training_args = GRPOConfig(
    output_dir="outputs/Llama-1B-base-GRPO",
    run_name="Llama-1B-base-GRPO-gsm8k",
    learning_rate=1e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.95,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=6,
    num_generations=12,
    max_completion_length=512,
    max_grad_norm=0.01,
    # report_to="wandb",
    log_on_each_node=False,
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

model_name = "meta-llama/Llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

# model = get_peft_model(model, peft_config)
# print(model.print_trainable_parameters())

tokenizer = AutoTokenizer.from_pretrained(model_name + '-Instruct')
tokenizer.pad_token = tokenizer.eos_token


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
    train_dataset=dataset,
    peft_config=peft_config,
)
trainer.train()