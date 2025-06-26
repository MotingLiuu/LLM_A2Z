# SFT-Mistral-7b-beta

## Run the expreriment

```bash
accelerate launch --config_file deepspeed_zero3.yaml run_sft.py config_full.yaml
```

## Config_files

**deepspeed_zero3.yaml**

```yam
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

**config_full.yaml**

```yaml
# Model arguments
model_name_or_path: mistralai/Mistral-7B-v0.1
model_revision: main
torch_dtype: bfloat16
attn_implementation: flash_attention_2

# Data training arguments
chat_template: "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
dataset_mixer:
  HuggingFaceH4/ultrachat_200k: 1.0
dataset_splits:
- train_sft
- test_sft
preprocessing_num_workers: 12

# SFT trainer config
bf16: true
do_eval: true
eval_strategy: epoch
gradient_accumulation_steps: 1
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: False
hub_model_id: zephyr-7b-sft-full
hub_strategy: every_save
learning_rate: 2.0e-05
log_level: info
logging_steps: 5  
logging_strategy: steps
lr_scheduler_type: cosine
max_seq_length: 2048
max_steps: -1
num_train_epochs: 1
output_dir: data/zephyr-7b-sft-full
overwrite_output_dir: true
per_device_eval_batch_size: 4
per_device_train_batch_size: 8
push_to_hub: true
remove_unused_columns: true
report_to:
- tensorboard
save_strategy: "steps"
save_steps: 100
save_total_limit: 1
seed: 42
warmup_ratio: 0.1
```



# SimPO

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml run_simpo.py mistral-7b-instruct-simpo.yaml
```

