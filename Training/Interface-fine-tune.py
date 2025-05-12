%%bash
pip install -qqq accelerate transformers auto-gptq optimum

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

set_seed(2024)

prompt = "Africa is an emerging economy because"

model_checkpoint = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                             trust_remote_code=True,
                                             torch_dtype="auto",
                                             device_map="cuda")

inputs = tokenizer(prompt,
                   return_tensors="pt").to("cuda")
outputs = model.generate(**inputs,
                         do_sample=True, max_new_tokens=120)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

set_seed(2024)

prompt = "Write a Python code that reads the content of multiple text files and save the result as CSV"

model_checkpoint = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                             trust_remote_code=True,
                                             torch_dtype="auto",
                                             device_map="cuda")

inputs = tokenizer(prompt,
                   return_tensors="pt").to("cuda")
outputs = model.generate(**inputs,
                         do_sample=True, max_new_tokens=200)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)



!pip install -qqq --upgrade transformers bitsandbytes accelerate datasets

%%bash
pip -q install huggingface_hub transformers peft bitsandbytes
pip -q install trl xformers
pip -q install datasets
pip install torch>=1.10

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from huggingface_hub import ModelCard, ModelCardData, HfApi
from datasets import load_dataset
from jinja2 import Template
from trl import SFTTrainer
import yaml
import torch

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
NEW_MODEL_NAME = "opus-samantha-phi-3-mini-4k"

DATASET_NAME = "macadeliccc/opus_samantha"
SPLIT = "train"
MAX_SEQ_LENGTH = 2048
num_train_epochs = 1
license = "apache-2.0"
username = "zoumana"
learning_rate = 1.41e-5
per_device_train_batch_size = 4
gradient_accumulation_steps = 1

if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
else:
  compute_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
dataset = load_dataset(DATASET_NAME, split="train")

EOS_TOKEN=tokenizer.eos_token_id

dataset

# Select a subset of the data for faster processing
dataset = dataset.select(range(100))

dataset

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    mapper = {"system": "system\n", "human": "\nuser\n", "gpt": "\nassistant\n"}
    end_mapper = {"system": "", "human": "", "gpt": ""}
    for convo in convos:
        text = "".join(f"{mapper[(turn := x['from'])]} {x['value']}\n{end_mapper[turn]}" for x in convo)
        texts.append(f"{text}{EOS_TOKEN}")
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
print(dataset['text'][8])



args = TrainingArguments(
    evaluation_strategy="steps",
    per_device_train_batch_size=7,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    max_steps=-1,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    output_dir=NEW_MODEL_NAME,
    optim="paged_adamw_32bit",
    lr_scheduler_type="linear"
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=128,
    formatting_func=formatting_prompts_func
)

"""
device = 'cuda'
import gc
import os
gc.collect()
torch.cuda.empty_cache()
"""

trainer.train()
