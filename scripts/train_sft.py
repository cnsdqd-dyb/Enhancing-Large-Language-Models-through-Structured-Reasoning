#!/usr/bin/env python3
"""Supervised fine-tuning on tagged Structured Reasoning traces."""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class DataArguments:
    dataset_name: str = field(default="FreeFrank/Structured-Reasoning")
    dataset_config: str | None = field(default=None)
    split: str = field(default="train")
    max_length: int = field(default=32768)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    trust_remote_code: bool = field(default=False)


def _messages(example: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    prompt = [{"role": "user", "content": example["problem"]}]
    target = (
        f"<think>\n{example['reasoning']}\n</think>\n"
        f"{example.get('content') or example.get('answer', '')}"
    )
    return prompt, prompt + [{"role": "assistant", "content": target}]


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example: dict) -> dict:
        prompt, conversation = _messages(example)
        prompt_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )
        input_ids = tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=False
        )[: data_args.max_length]
        prompt_length = min(len(prompt_ids), len(input_ids))
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * prompt_length + input_ids[prompt_length:],
        }

    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config,
        split=data_args.split,
    )
    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
