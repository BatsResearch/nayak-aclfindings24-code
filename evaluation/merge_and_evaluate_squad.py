# code for merge and evaluate squadshifts
# in this code we will load the qlora weights and the base model.
# we will then merge the weights, if necessary, and evaluate on squadshifts.

import argparse
import json
import logging
import os
import random
import re
import shutil
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from datasets import load_dataset
from evaluate import load
from peft import PeftModel
from promptsource.templates import DatasetTemplates
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_and_merge_model(base_model, adapter_path):
    """
    Function loads the adapter to the base model and merges the weights in the base
    model and returns the model.
    """
    logger.info(f"Loading the base model {base_model} into memory")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if adapter_path is not None:
        logger.info(f"Loading the adapter model {adapter_path} into memory")
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Merging the weights")
        model = model.merge_and_unload()

    return model


def save_model(model, tokenizer, save_path):
    """
    Save the model and tokenizer to the save path
    (preferrably the scratch path)
    """
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def preprocess_model(base_model, adapter_path, scratch_path):
    # load and merge the model
    model = load_and_merge_model(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # save the model
    logger.info("Saving the model to scratch")
    model_name = os.path.basename(os.path.dirname(adapter_path.rstrip("/")))
    logger.info(f"Model name: {model_name}")
    save_path = os.path.join(scratch_path, model_name)
    save_model(model, tokenizer, save_path)

    assert os.path.exists(save_path)

    return save_path


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    checkpoint_model_id_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    scratch_path: Optional[str] = field(
        default="~/scratch",
        metadata={"help": "The path to the scratch directory to save the model."},
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."},
    )


@dataclass
class TestArguments:
    output_dir: str = field(
        default=None, metadata={"help": "output dir to save the results"}
    )
    dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library).",
        },
    )
    simple_prompt: bool = field(
        default=False, metadata={"help": "whether to use simple prompts or not."}
    )


def evaluate_exqa():
    parser = HfArgumentParser((ModelArguments, TestArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, test_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, test_args = parser.parse_args_into_dataclasses()

    args = argparse.Namespace(**vars(model_args), **vars(test_args))

    scratch_path = os.path.expanduser("~/scratch")

    # get model path
    if args.checkpoint_model_id_or_path is None:
        model_path = args.model_name_or_path
    else:
        model_path = preprocess_model(
            args.model_name_or_path, args.checkpoint_model_id_or_path, scratch_path
        )

    model = LLM(model=model_path, trust_remote_code=True)

    # get the relevant templates
    template_names = [
        "after",
        "exam",
        "wondered",
        "pick_one_answer",
        "question_num_hint_answer",
    ]

    if args.dataset_name == "reddit":
        dataset_templates = DatasetTemplates("squadshifts", "amazon")
    else:
        dataset_templates = DatasetTemplates("squadshifts", args.dataset_name)

    templates = [(name, dataset_templates[name]) for name in template_names]

    context = "{{context}}"

    modified_dataset_name = f"squadshifts_{args.dataset_name}"
    output_path = os.path.join(args.output_dir, modified_dataset_name)
    os.makedirs(output_path, exist_ok=True)

    # iterate over all the templates and save the results in the output directory
    for template_name, template in templates:
        dataset = load_dataset("squadshifts", args.dataset_config_name)["test"]
        column_names = dataset.column_names

        if args.dataset_config_name == "reddit":
            template.jinja = re.sub("Amazon", "Reddit", template.jinja)

        if test_args.use_bonito:
            jinja_template = re.sub(
                re.escape(context), "{{copy_context}}", template.jinja
            )
            jinja_template = (
                f"<|tasktype|>\nextractive question answering\n<|context|>\n{context.strip()}\n<|task|>\n"
                + jinja_template
            )
            dataset = dataset.add_column("copy_context", ["{{context}}"] * len(dataset))
            template.jinja = jinja_template
            column_names = dataset.column_names

        def preprocess_squad_batch(examples):
            input = []
            for i in range(len(examples["context"])):
                ex = {k: examples[k][i] for k in column_names}
                source, target = template.apply(ex)
                input.append(source)

            label = []
            for i in range(len(examples["context"])):
                label.append(examples["answers"][i]["text"][0])

            return {
                "input": input,
                "label": label,
            }

        def add_instruct_template(examples):
            bs = len(examples["input"])
            inputs = []
            for i in range(bs):
                input_text = (
                    "<|input|>\n" + examples["input"][i].strip() + "\n<|output|>\n"
                )
                inputs.append(input_text)

            return {
                "input": inputs,
            }

        def add_mistral_instruct_template(examples):
            bs = len(examples["input"])
            inputs = []
            for i in range(bs):
                input_text = "[INST] " + examples["input"][i].strip() + " [/INST]"
                inputs.append(input_text)

            return {
                "input": inputs,
            }

        def add_simple_prompts(examples):
            bs = len(examples["input"])
            inputs = []
            for i in range(bs):
                input_text = "Instruct: " + examples["input"][i].strip() + "\nAnswer:"
                inputs.append(input_text)

            return {
                "input": inputs,
            }

        dataset = dataset.map(
            preprocess_squad_batch,
            remove_columns=column_names,
            batched=True,
        )

        if args.checkpoint_model_id_or_path is None and (
            args.model_name_or_path == "mistralai/Mistral-7B-v0.1"
            or args.model_name_or_path == "meta-llama/Llama-2-7b-hf"
        ):
            args.simple_prompt = True

        if args.simple_prompt:
            dataset = dataset.map(
                add_simple_prompts,
                batched=True,
            )
        elif args.model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
            dataset = dataset.map(
                add_mistral_instruct_template,
                batched=True,
            )
        else:
            dataset = dataset.map(
                add_instruct_template,
                batched=True,
            )
        metric = load("squad")

        sampling_params = SamplingParams(temperature=0, max_tokens=25)
        print(dataset["input"][:2])
        predictions = model.generate(dataset["input"], sampling_params=sampling_params)
        _predictions = [p.outputs[0].text.strip() for p in predictions]

        p = [{"id": str(i), "prediction_text": p} for i, p in enumerate(_predictions)]
        r = [
            {"id": str(i), "answers": {"text": [l], "answer_start": [0]}}
            for i, l in enumerate(dataset["label"])
        ]

        result = metric.compute(predictions=p, references=r)
        print(result)
        template_path = os.path.join(output_path, f"results_{template_name}.json")
        with open(template_path, "w+") as fp:
            json.dump(result, fp)

    if args.delete_scratch:
        logger.info("deleting the model directory from scratch")
        shutil.rmtree(model_path)


if __name__ == "__main__":
    evaluate_exqa()

    logger.info("Done!")
