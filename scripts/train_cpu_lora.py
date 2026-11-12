#!/usr/bin/env python3
"""
CPU & Low-RAM Friendly LoRA Fine-Tuning Script.
Optimized for 0.5B - 1.7B models (Qwen2.5-0.5B, SmolLM2-1.7B, Llama-3.2-1B) on CPU or single GPU.
"""
import argparse
import json
from pathlib import Path
from typing import Any
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def load_jsonl_messages(path: str, tokenizer: Any, max_length: int) -> Any:
    texts: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj: dict[str, Any] = json.loads(line)
            msgs: list[dict[str, Any]] = obj.get("messages", [])
            try:
                rendered: str = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception:
                rendered = ""
                for m in msgs:
                    role: str = str(m.get("role", "user"))
                    content: str = str(m.get("content", ""))
                    if "tool_calls" in m:
                        content += "\n<tool_call>\n" + json.dumps(m["tool_calls"]) + "\n</tool_call>"
                    rendered += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            texts.append({"text": rendered})

    ds = Dataset.from_list(texts)

    def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")

    return ds.map(tokenize_fn, batched=True, remove_columns=["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a small model with LoRA on CPU/Low-RAM.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF Model ID (<2B recommended for CPU)")
    parser.add_argument("--dataset", required=True, help="Path to validated train_traces.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory to save LoRA adapter and merged model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--merge", action="store_true", help="Merge LoRA into base weights for Ollama/GGUF import")
    args = parser.parse_args()

    use_cuda: bool = torch.cuda.is_available()
    device_map: str = "auto" if use_cuda else "cpu"
    dtype: torch.dtype = torch.float16 if use_cuda else torch.float32

    print(f"[INFO] Loading {args.base_model} on {device_map} ({dtype})...")
    tokenizer: Any = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: Any = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds: Any = load_jsonl_messages(args.dataset, tokenizer, args.max_seq_len)

    adapter_dir = Path(args.output_dir) / "adapter"
    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        use_cpu=not use_cuda,
        fp16=use_cuda,
        dataloader_pin_memory=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("[INFO] Starting LoRA training...")
    trainer.train()
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[PASS] Saved LoRA adapter to {adapter_dir}")

    if args.merge:
        merged_dir = Path(args.output_dir) / "merged_hf"
        print(f"[INFO] Merging LoRA adapter into base model at {merged_dir} for Ollama/GGUF...")
        merged_model: Any = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        print(f"[PASS] Merged model ready at {merged_dir}")


if __name__ == "__main__":
    main()
