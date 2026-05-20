"""
VulnChain SFT Training Script
Fine-tunes a 7B model on exploit chain reasoning traces using Unsloth + TRL.

Usage:
    python training/sft_train.py --config training/configs/sft_config.yaml
    python training/sft_train.py --config training/configs/sft_config.yaml --debug  # 10-step sanity check
"""

import argparse
import yaml
from pathlib import Path

import wandb
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        dtype=None,  # Auto-detect
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template="mistral")
    return model, tokenizer


def format_sample(sample: dict, tokenizer) -> dict:
    """
    Applies chat template to instruction/input/output format.
    Adjust based on your data schema.
    """
    messages = [
        {"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"},
        {"role": "assistant", "content": sample["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def main(config_path: str, debug: bool = False):
    cfg = load_config(config_path)

    # Init W&B
    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        tags=cfg["wandb"]["tags"],
        config=cfg,
    )

    # Build model
    print(f"Loading model: {cfg['model']['name']}")
    model, tokenizer = build_model(cfg)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["data"]["train_file"],
            "eval": cfg["data"]["eval_file"],
        },
    )

    if debug:
        print("DEBUG MODE: using 10 samples only")
        dataset["train"] = dataset["train"].select(range(10))
        dataset["eval"] = dataset["eval"].select(range(5))

    if cfg["data"]["max_samples"]:
        dataset["train"] = dataset["train"].select(
            range(min(cfg["data"]["max_samples"], len(dataset["train"])))
        )

    # Apply chat template
    dataset = dataset.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    # Training config
    training_args = SFTConfig(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"] if not debug else 1,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_steps=cfg["training"]["eval_steps"] if not debug else 5,
        save_steps=cfg["training"]["save_steps"] if not debug else 10,
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        metric_for_best_model=cfg["training"]["metric_for_best_model"],
        max_steps=10 if debug else -1,
        report_to=cfg["training"]["report_to"],
        dataset_text_field="text",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=training_args,
    )

    print("Starting SFT training...")
    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"] + "_final")
    print(f"Model saved to {cfg['training']['output_dir']}_final")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config, args.debug)
