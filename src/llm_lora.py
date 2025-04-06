# -*- coding: utf-8 -*-

# import os
# !pip install unsloth
# # Install latest Hugging Face for Gemma-3!
# !pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft "trl==0.15.2" triton cut_cross_entropy unsloth_zoo
# !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer

from unsloth import FastModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import TextStreamer

max_seq_length = 4096  # Up to 32000. Auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # 4 bit quantization to reduce memory
    load_in_8bit=False,  # A bit more accurate, uses 2x memory
    full_finetuning=False,
)
"""We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # Turn off for just text!
    finetune_language_layers=True,  # Should leave on!
    finetune_attention_modules=True,  # Attention good for GRPO
    finetune_mlp_modules=True,  # SHould leave on always!
    r=8,  # Larger = higher accuracy, but might overfit
    lora_alpha=8,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


dataset = load_dataset("json", data_files="train.jsonl")
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # max_steps=60, # to speed things up
        max_steps=None,  # for a full run
        num_train_epochs=2,  # for a full run
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
                         3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(
    f"Peak reserved memory for training % of max memory = {lora_percentage} %."
)
### Inference

line25 = {
    "instruction":
    "Please identify several topic boundaries for the following text. Each topic consists of several consecutive sentences. Please output in the form of a list: [ hash_1, hash_2, ..., hash_n ], where the elements in the list are the hashes of the sentences that separate different topics.",
    "input":
    "The effects of the disease can have profound effects on everyday life. sentence_hash: f58206 As well, the recurring side effects of excessive belching, dizziness, dry mouth, hangovers, disorientation, irritable bowel syndrome, and chronic fatigue syndrome can lead to other health problems such as depression, anxiety and poor productivity in employment. sentence_hash: 3e044f The random state of intoxication can lead to personal difficulties, and the relative obscurity of the condition can also make it hard to seek treatment. sentence_hash: 6d8cfd The treatment for auto-brewery syndrome is a change in diet requiring low carbohydrates and high protein. sentence_hash: 2a323e Sugar is fermented into alcohol, and a diet that effectively lowers sugars also lowers the alcohol that can be fermented from it. sentence_hash: 6802d5 Anything that causes an imbalance between the beneficial and harmful bacteria in the gut can help increase the chance that fermentation in the gut will develop. sentence_hash: a834b4 This can include not only antibiotics, but also overindulgence in sugars and carbohydrates. sentence_hash: 8fa124 Watching what you eat could lower the risk of gut fermentation syndrome, and taking probiotics could further protect you by increasing the number of good bacteria in your system. sentence_hash: fb590a",
    "output": "[ 2a323e ]"
}

# alpaca_prompt = Copied from above
FastModel.for_inference(model)  # Enable native 2x faster inference
inputs = tokenizer(
    [
        alpaca_prompt.format(
            line25["instruction"],  # instruction
            line25["input"],  # input
            "",  # output - leave this blank for generation!
        )
    ],
    return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
tokenizer.batch_decode(outputs)
""" You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!"""

# alpaca_prompt = Copied from above
FastModel.for_inference(model)  # Enable native 2x faster inference
inputs = tokenizer(
    [
        alpaca_prompt.format(
            line25["instruction"],  # instruction
            line25["input"],  # input
            "",  # output - leave this blank for generation!
        )
    ],
    return_tensors="pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

model.save_pretrained("lora_model")  # Local saving
