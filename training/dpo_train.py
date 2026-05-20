"""
VulnChain DPO Training Script
Aligns SFT model using Direct Preference Optimization with rule-based verifier signals.

Usage:
    python training/dpo_train.py --config training/configs/dpo_config.yaml
"""

import argparse
import yaml

import wandb
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        tags=cfg["wandb"]["tags"],
        config=cfg,
    )

    # Load SFT checkpoint
    print(f"Loading SFT checkpoint: {cfg['model']['sft_checkpoint']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["sft_checkpoint"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    # Load preference data
    # Expected format: {"prompt": str, "chosen": str, "rejected": str}
    dataset = load_dataset("json", data_files=cfg["data"]["preference_file"])["train"]
    dataset = dataset.train_test_split(test_size=0.1)

    training_args = DPOConfig(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        bf16=cfg["training"]["bf16"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_steps=cfg["training"]["eval_steps"],
        save_steps=cfg["training"]["save_steps"],
        report_to=cfg["training"]["report_to"],
        beta=cfg["dpo"]["beta"],
        loss_type=cfg["dpo"]["loss_type"],
        max_prompt_length=cfg["dpo"]["max_prompt_length"],
        max_length=cfg["dpo"]["max_length"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth handles reference model automatically
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"] + "_final")
    print(f"DPO model saved to {cfg['training']['output_dir']}_final")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
