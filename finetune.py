from collections import defaultdict
from functools import partial
import os
import sqlite3
import yaml

import dotenv

dotenv.load_dotenv()

import torch
from torch import nn

import bitsandbytes as bnb
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import tqdm as tqdm

import llm_planner as llmp

from accelerate import Accelerator


HF_USER_TOKEN = os.getenv("HF_USER_TOKEN")


def load_dataset(config: dict) -> dict[str, Dataset]:
    """Load the dataset from the configuration.

    Args:
        config (dict): The dataset configuration.

    Returns:
        dict[str, Dataset]: The loaded dataset.
    """
    with open(config["splits_path"], "r") as f:
        split_ids_cfg = yaml.safe_load(f)

    splits: set[str] = config.get("splits", {}).keys()
    dataset = {split: defaultdict(list) for split in splits}

    # Connect to database
    conn = sqlite3.connect(config["database_path"])
    c = conn.cursor()

    # load domains
    domains = {}
    c.execute("SELECT name, domain_pddl FROM domains")
    for domain_name, domain_pddl in c.fetchall():
        domains[domain_name] = domain_pddl

    # load problems
    for split in splits:
        queries = []
        split_keys: list[str] = config["splits"][split]
        for split_key in split_keys:
            split_ids = split_ids_cfg
            for key in split_key:
                split_ids = split_ids[key]

            c.execute(
                f"SELECT domain, problem_pddl, natural_language FROM problems WHERE id in ({', '.join(['?'] * len(split_ids))})",
                split_ids,
            )
            queries.extend(c.fetchall())

        for domain, problem_pddl, natural_language in queries:
            dataset[split]["domain"].append(domains[domain])
            dataset[split]["problem"].append(problem_pddl)
            dataset[split]["natural_language"].append(natural_language)

    return {s: Dataset.from_dict(d, split=s) for s, d in dataset.items()}


def find_all_linear_names(
    model: nn.Module,
    bits: int | None = None,
) -> list[str]:
    """Find names of all linear layers in the model.

    Args:
        model (nn.Module): The model to search for linear layers.

    Returns:
        list[str]: The names of all linear layers in the model (excluding LM Head)
    """
    match bits:
        case 4:
            Linear = bnb.nn.Linear4bit
        case 8:
            Linear = bnb.nn.Linear8bitLt
        case _:
            Linear = torch.nn.Linear

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            names = name.split(".")
            lora_module_names.add(names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def strip(text: str, bos_token: str, eos_token: str) -> str:
    return text.removeprefix(bos_token) + eos_token


def preprocess(
    tokenizer: PreTrainedTokenizer,
    examples,
    domain_prompt: str = "",
    problem_prompt: str = "",
) -> list[str]:
    """Preprocess the examples for training.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        examples: The examples to preprocess.
        domain_prompt (str, optional): How to prompt the domain. Defaults to "".
        problem_prompt (str, optional): How to prompt the problem. Defaults to "".

    Returns:
        list[str]: The preprocessed examples.
    """
    inputs = [
        strip(
            tokenizer.apply_chat_template(
                    llmp.PlanningProblem(nl, d, p).apply_template(
                    domain_prompt,
                    problem_prompt,
                ),
                tokenize=False,
                add_generation_prompt=False,
            ),
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
        )
        for nl, d, p in zip(
            examples["natural_language"],
            examples["domain"],
            examples["problem"],
        )
    ]
    return inputs


def load_model(config: dict) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load the model and tokenizer from the configuration.

    Args:
        config (dict): The training config.

    Returns:
        tuple[PreTrainedTokenizer, PreTrainedModel]: The tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["tokenizer_name"],
        token=HF_USER_TOKEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config_args: dict = config.get("bnb_config", {})
    if bnb_config_args:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bnb_config_args.get("bits", 16) == 4,
            load_in_8bit=bnb_config_args.get("bits", 16) == 8,
            bnb_4bit_use_double_quant=bnb_config_args.get("use_double_quant", False),
            bnb_4bit_quant_type=bnb_config_args.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    device_index = Accelerator().process_index
    device_map = {"": device_index}
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_name"],
        **config["model"].get("model_kwargs", {}),
        token=HF_USER_TOKEN,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map=device_map,
    )

    lora_config = LoraConfig(
        **config["lora_config"],
        target_modules=find_all_linear_names(model, bits=bnb_config_args.get("bits")),
    )
    model = get_peft_model(model, lora_config)

    return tokenizer, model


def main(config_path: str):
    """Train a model on a dataset using a given configuration.

    Args:
        config_path (str): The path to the configuration file.
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load dataset
    dataset = load_dataset(config["dataset"])

    train_config = config["train"]

    # Load model
    tokenizer, model = load_model(train_config)

    # Create data collator
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer.encode(
            train_config["model"]["response_template"],
            add_special_tokens=False,
        ),
        tokenizer=tokenizer,
    )

    # Build training arguments
    args_config = train_config.get("training_args", {})
    training_args = TrainingArguments(**args_config)

    # Create trainer
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        max_seq_length=train_config["model"].get("max_seq_length", 512),
        formatting_func=partial(
            preprocess,
            tokenizer,
            problem_prompt=config["dataset"]["prompts"]["problem"],
            domain_prompt=config["dataset"]["prompts"]["domain"],
        ),
    )
    trainer.train()

    trainer.save_model(train_config.get("save_path", "ckpt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Fine-tune a model on PDDL dataset.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        required=True,
        help="Path to the configuration file.",
    )

    args = parser.parse_args()

    main(args.config)
