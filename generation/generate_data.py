import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict
import logging

import torch
import transformers
import argparse
from transformers import (
    set_seed,
)
from datasets import load_dataset, Dataset
from bonito import Bonito, SamplingParams


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TASKS_DICTS = {
    "exqa": "extractive question answering",
    "mcqa": "multiple-choice question answering",
    "qg": "question generation",
    "qa": "question answering without choices",
    "ynqa": "yes-no question answering",
    "coref": "coreference resolution",
    "paraphrase": "paraphrase generation",
    "paraphrase_id": "paraphrase identification",
    "sent_comp": "sentence completion",
    "sentiment": "sentiment",
    "summarization": "summarization",
    "text_gen": "text generation",
    "topic_class": "topic classification",
    "wsd": "word sense disambiguation",
    "te": "textual entailment",
    "nli": "natural language inference",
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The directory of the QLoRA checkpoint to use the generation."
        },
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    use_auth_token: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."},
    )
    enforce_eager: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables eager execution for the model."},
    )


@dataclass
class DataArguments:
    output_dir: str = field(
        default=None,
        metadata={
            "help": "The output directory where the model predictions will be written."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "Which dataset to finetune on. See datamodule for options."},
    )
    text_dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to a local text dataset. This should be the huggingface dataset format."
        },
    )
    scratch_path: str = field(
        default=None,
        metadata={"help": "Path to the scratch directory to save the model."},
    )
    task_type: Optional[str] = field(
        default=None,
        metadata={"help": "Which task types to train on. [paragraph|sentence|all]"},
    )
    shuffle: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to shuffle the dataset before selecting samples."},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Seed for random number generator."},
    )
    generation_type: Optional[str] = field(
        default="bonito",
        metadata={"help": "Which generation type to use. [bonito|zephyr]"},
    )
    start_index: Optional[int] = field(
        default=None,
        metadata={"help": "Which index to start generating from."},
    )
    end_index: Optional[int] = field(
        default=None,
        metadata={"help": "Which index to end generating from."},
    )


@dataclass
class SamplingParameters:
    n: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of samples to generate for each input.",
        },
    )
    temperature: Optional[float] = field(
        default=0.95,
        metadata={
            "help": "Temperature for sampling. ",
        },
    )
    top_p: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.",
        },
    )
    max_tokens: Optional[int] = field(
        default=256,
        metadata={
            "help": "Maximum number of tokens to generate for each sample.",
        },
    )


def generate_with_model(context_dataset, args, sampling_args):
    # load vllm model
    if args.task_type:
        task_type = args.task_type
    else:
        # get the default task type for the dataset
        task_to_type = {
            "pubmed_qa": "ynqa",
            "privacy_qa": "ynqa",
            "squadshifts_nyt": "exqa",
            "squadshifts_amazon": "exqa",
            "squadshifts_reddit": "exqa",
            "contract_nli": "nli",
            "vitaminc": "nli",
        }
        task_type = task_to_type[args.dataset_name]

    n_gpus = torch.cuda.device_count()
    llm = Bonito()
    sampling_params = SamplingParams(**vars(sampling_args))

    # show example from the dataset
    logger.info("Showing an example from the dataset")
    logger.info(context_dataset["context"][0])

    os.makedirs(args.output_dir, exist_ok=True)

    # generate the output
    logger.info("Generating instruction tuning dataset")
    generate_instruction_tuning_dataset(
        context_dataset=context_dataset,
        generation_type=args.generation_type,
        task_type=task_type,
        sampling_params=sampling_params,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        model=llm,
        start_index=args.start_index,
        end_index=args.end_index,
    )


def generate_instruction_tuning_dataset(
    context_dataset,
    generation_type,
    task_type,
    sampling_params,
    dataset_name,
    output_dir,
    model=None,
    **kwargs,
):
    """
    Generate text from a context using a Bonito model
    """
    if generation_type == "bonito":
        bonito_generate(
            context_dataset=context_dataset,
            task_type=task_type,
            model=model,
            sampling_params=sampling_params,
            dataset_name=dataset_name,
            output_dir=output_dir,
            **kwargs,
        )
    else:
        NotImplementedError("Invalid generation type")


def bonito_generate(
    context_dataset,
    task_type,
    model,
    sampling_params,
    dataset_name,
    output_dir,
    start_index=None,
    end_index=None,
    **kwargs,
):
    """
    Generate text from a context using a Bonito model
    """
    synthetic_dataset = model.generate_tasks(
        context_dataset,
        context_col="context",
        task_type=task_type,
        sampling_params=sampling_params,
    )

    if start_index is not None and end_index is not None:
        synthetic_dataset.save_to_disk(
            os.path.join(output_dir, f"bonito_{dataset_name}_{start_index}_{end_index}")
        )
    else:
        synthetic_dataset.save_to_disk(
            os.path.join(output_dir, f"bonito_{dataset_name}")
        )


def load_context(
    dataset_name,
    max_eval_samples=None,
    shuffle=False,
    start_index=None,
    end_index=None,
):
    _dataset_name = "unannotated_{}".format(dataset_name)
    split = "train"
    dataset = load_dataset("BatsResearch/bonito-experiment", _dataset_name)[split]

    # rename input to context
    dataset = dataset.rename_column("input", "context")

    # context.strip()
    dataset = dataset.map(
        lambda example: {
            "context": example["context"].strip(),
        },
        remove_columns=dataset.column_names,
    )

    #
    if max_eval_samples is not None and len(dataset) > max_eval_samples:
        if shuffle:
            dataset = dataset.shuffle().select(range(max_eval_samples))
        else:
            dataset = dataset.select(range(max_eval_samples))

    if start_index is not None and end_index is not None:
        # check if the start index is less than the size of the dataset
        assert start_index < len(dataset)
        if end_index > len(dataset):
            end_index = len(dataset)

        logger.info(
            "slicing up the dataset from {} to {}".format(start_index, end_index)
        )
        dataset = dataset.select(range(start_index, end_index))

    return dataset


def generate():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, SamplingParameters)
    )
    (
        model_args,
        data_args,
        sampling_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(sampling_args)
    )
    print(args)
    set_seed(args.seed)

    # get the contexts
    context_dataset = load_context(
        args.dataset_name,
        args.max_eval_samples,
        args.shuffle,
        args.start_index,
        args.end_index,
    )

    # show example from the dataset
    logger.info("Showing an example from the dataset")
    logger.info(context_dataset["context"][0])

    #
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Using model to generate")
    generate_with_model(context_dataset, args, sampling_args)

    # save args
    with open(os.path.join(args.output_dir, "args.json"), "w+") as fout:
        json.dump(vars(args), fout)

    logger.info("Done!")


if __name__ == "__main__":
    generate()
