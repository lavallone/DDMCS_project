import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer
import warnings
import argparse

def finetune(dataset_type:str):
    
    model_name = "microsoft/Phi-3-mini-128k-instruct"
    output_dir = f"../checkpoints/{dataset_type}/"

    ## prepare DATASET
    dataset_path = f"../../data/sentiment_analysis/finetuning/training_{dataset_type}.json"
    data = load_dataset("json", data_files=dataset_path)
    data = data["train"].train_test_split(test_size=0.1)

    ## prepare TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    ## prepare MODEL
    # quantization step
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # setup lora configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    )
    peft_model = get_peft_model(model, peft_config)

    ## TRAIN
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        push_to_hub=False,
        num_train_epochs=10,
        save_total_limit=1,
        evaluation_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        peft_config=peft_config,
        dataset_text_field="prompt",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    peft_model.config.use_cache = False

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.bfloat16)

    trainer.train()
    #trainer.push_to_hub()

def generate(dataset_type:str):
    pass

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    supported_modes = ["finetune", "generate"]
    supported_dataset_types = ["emotion", "sentiment"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, help="Input the mode.")
    parser.add_argument("--dataset_type", "-dt", type=str, help="Input the datset type.")
    args = parser.parse_args()
    
    if args.mode == "finetune":
        finetune(args.dataset_type) 
    else:
        generate(args.dataset_type)
