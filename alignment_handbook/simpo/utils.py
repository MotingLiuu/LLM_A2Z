from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import BatchEncoding
from transformers import AutoModel
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.stats import pearsonr
from tqdm import tqdm
import torch
import os
from multiprocessing import Process, Queue, set_start_method
device = "cuda" if torch.cuda.is_available() else "cpu"
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
set_start_method('spawn', force=True)
logger.info("Multiprocessing start method set to 'spawn'.")

def _run_model_inference_worker(
    model: nn.Module,
    prompt: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    response_queue: Queue,
    max_new_tokens: int = 100,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 0.7,
    ):
    try:
        prompt = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id
            )
            logger.debug(f"generated_ids: {generated_ids}, prompt: {prompt.input_ids}")
            response_ids = generated_ids[:, prompt.input_ids.shape[1]:]
            response_queue.put((model.name_or_path, response_ids))
                            
    except Exception as e:
        logger.error(f"Error during inference: {e}, model: {model.name_or_path}")
        response_queue.put((None, str(e)))

def run_model_inference(gpu_id, model_name, alias, prompt, num_generations, request_queue, response_queue):
    try:
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Process {os.getpid()}: Loading model '{alias}' ({model_name}) on {device}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch_dtype,
            device_map=device, 
            trust_remote_code=True,
        )

        logger.info(f"Process {os.getpid()}: Model '{alias}' loaded successfully and is on device: {model.device}")

        log_likelihoods = []
        response_lengths = []

        prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        prompt_length = prompt_inputs.input_ids.shape[1]

        with torch.no_grad():
            for _ in tqdm(range(num_generations), desc=f"Generating with {alias}"):
                max_len = np.random.randint(20, 200)
                generated_ids = model.generate(
                    **prompt_inputs,
                    max_new_tokens=max_len,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )

                if generated_ids.shape[1] <= prompt_length:
                    continue

                response_ids = generated_ids[:, prompt_length:]
                response_length = response_ids.shape[1]

                if response_length == 0: 
                    continue

                full_ids = generated_ids 
                outputs = model(full_ids, labels=full_ids) 
                logits = outputs.logits 

                logits_for_response = logits[:, prompt_length - 1 : -1, :] 
                
                if logits_for_response.shape[1] != response_ids.shape[1]:
            
                    log_probs_full = torch.nn.functional.log_softmax(logits, dim=-1)
                    true_token_log_probs_full = log_probs_full.gather(
                        dim=-1, index=generated_ids.unsqueeze(-1)
                    ).squeeze(-1) 

                    avg_log_prob = true_token_log_probs_full[:, prompt_length:].mean().item()

                else: 
                    log_probs = torch.nn.functional.log_softmax(logits_for_response, dim=-1)
                    true_token_log_probs = torch.gather(log_probs, 2, response_ids.unsqueeze(-1)).squeeze(-1)
                    avg_log_prob = true_token_log_probs.mean().item()

                response_lengths.append(response_length)
                log_likelihoods.append(avg_log_prob)

        response_queue.put((alias, log_likelihoods, response_lengths))

    except Exception as e:
        logger.error(f"Process {os.getpid()}: Error in model '{alias}': {e}")
        response_queue.put((alias, None, str(e))) 