import os
import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOConfig, DPOTrainer


log_file = "dpo_full_finetune_training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting DPO full fine-tuning training script...")


# --- 2. 加载数据集 ---
logger.info("Loading UltraFeedback binarized dataset...")
# 在你的环境中，你可以使用本地缓存目录
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", cache_dir="./data/ultrafeedback_binarize")

# 将数据集拆分为训练集和验证集
dataset = dataset.train_test_split(test_size=0.01)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
logger.info(f"Dataset loaded. Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")


# --- 3. 加载模型和分词器 ---
model_name = "meta-llama/Llama-3.2-3B-Instruct"
logger.info(f"Loading base model for full fine-tuning: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", # 自动将模型分片到可用的GPU上
    torch_dtype=torch.bfloat16, # 使用bfloat16以获得更好的性能
)
model.config.use_cache = False
logger.info("Base model loaded for full fine-tuning.")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
logger.info("Tokenizer loaded.")


# --- 4. 配置训练参数 ---
# DPOTrainer使用DPOConfig来设置所有训练相关的参数
output_dir = "./dpo_llama2_ultrafeedback_full_finetune"
training_args = DPOConfig(
    output_dir=output_dir,                # 模型检查点和日志的输出目录
    num_train_epochs=1,                   # 训练周期数
    per_device_train_batch_size=1,        # 全参数微调时，batch_size需要设置得非常小
    per_device_eval_batch_size=1,         # 评估的batch_size
    gradient_accumulation_steps=16,       # 增大了梯度累积以模拟更大的batch_size (1*16=16)
    gradient_checkpointing=True,          # 使用梯度检查点以节省内存
    learning_rate=5e-6,                   # 全参数微调时通常使用更小的学习率
    lr_scheduler_type="cosine",           # 学习率调度器类型
    warmup_steps=100,                     # 预热步数
    optim="adamw_torch",                  # 标准的AdamW优化器
    bf16=True,                            # 使用bfloat16进行训练
    logging_strategy="steps",             # 按步数记录日志
    logging_steps=25,                     # 每隔25步记录一次日志
    eval_strategy="steps",                # 在训练过程中按步数进行评估
    eval_steps=200,                       # 每隔200步进行一次评估
    save_strategy="steps",                # 按步数保存模型检查点
    save_steps=200,                       # 每隔200步保存一次模型
    save_total_limit=2,                   # 最多保存2个检查点
    remove_unused_columns=False,          # 确保不移除DPO所需的列
    beta=0.1,                             # DPO损失中的beta参数
    report_to="none",                     # 禁用所有报告（如wandb）
)
logger.info(f"DPOConfig set up for full fine-tuning. Output directory: {output_dir}")

# --- 5. 创建DPOTrainer ---
# 注意：我们移除了 peft_config 参数
trainer = DPOTrainer(
    model,
    ref_model=None, # 我们让trainer自动处理参考模型
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)
logger.info("DPOTrainer initialized for full fine-tuning.")

# --- 6. 开始训练 ---
logger.info("Starting DPO training...")
trainer.train()
logger.info("Training finished.")

# --- 7. 保存最终模型 ---
final_model_path = os.path.join(output_dir, "final_model")
trainer.save_model(final_model_path)
logger.info(f"Final model saved to {final_model_path}")