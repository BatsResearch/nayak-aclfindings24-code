import argparse
import copy
import random
import re
from typing import Dict, Union
import numpy as np
import pandas as pd
from datasets import concatenate_datasets, DatasetDict, load_dataset
from promptsource.templates import DatasetTemplates
from transformers import set_seed
import os
from transformers import AutoTokenizer
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

set_seed(42)

dataset_context_keys = [
    ("cosmos_qa", None, "{{ context }}"),
    ("social_i_qa", None, "{{context}}"),
    ("paws", "labeled_final", "{{sentence1}}"),
    ("quail", None, "{{ context }}"),
    ("squad", None, "{{context}}"),
    ("dream", None, '{{dialogue | join("\n\n")}}'),
    ("qasc", None, "{{ fact1[0]|capitalize }}{{ fact1[1:]|trim|trim('.') }}"),
    (
        "wiki_hop",
        "original",
        "{% for support in supports %}\n- {{ support }}\n{% endfor %}",
    ),
    ("race", "all", "{{article}}"),
    ("hellaswag", None, "{{ctx}}"),
    ("super_glue", "boolq", "{{passage}}"),
    ("adversarial_qa", "dbert", "{{context}}"),
    ("adversarial_qa", "dbidaf", "{{context}}"),
    ("adversarial_qa", "droberta", "{{context}}"),
    ("quoref", None, "{{context}}"),
    ("duorc", "ParaphraseRC", "{{plot}}"),
    ("duorc", "SelfRC", "{{plot}}"),
    ("ropes", None, "{{ background }}"),
    ("super_glue", "record", "{{passage}}"),
    ("amazon_polarity", None, "{{content}}"),
    ("app_reviews", None, "{{review}}"),
    ("imdb", None, "{{text}}"),
    ("rotten_tomatoes", None, "{{text}}"),
    ("yelp_review_full", None, "{{text}}"),
    ("cnn_dailymail", "3.0.0", "{{article}}"),
    ("gigaword", None, "{{document}}"),
    ("multi_news", None, "{{document}}"),
    ("samsum", None, "{{dialogue}}"),
    ("xsum", None, "{{document}}"),
    ("ag_news", None, "{{text}}"),
    ("dbpedia_14", None, "{{content}}"),
    ("glue", "mrpc", "{{sentence1}}"),
    ("super_glue", "wsc.fixed", "{{text}}"),
    ("super_glue", "wic", "{{sentence1}}"),
    ("super_glue", "copa", "{{premise}}"),
    ("super_glue", "rte", "{{premise}}"),
    ("super_glue", "cb", "{{premise}}"),
    ("anli", None, "{{premise}}"),
    ("quartz", None, "{{ para }}"),
]


def get_filtered_template(
    dataset_name,
    dataset_config_name,
    context,
):
    dataset_templates = DatasetTemplates(dataset_name, dataset_config_name)
    templates = [template for template in dataset_templates.templates.values()]
    filtered_templates = []
    reverse_filtered_templates = []
    for i in range(len(templates)):
        t = copy.deepcopy(templates[i])

        jinja_template = t.jinja

        if dataset_name == "qasc":
            if context not in jinja_template:
                continue

        if context in jinja_template or re.escape(context) in jinja_template:
            jinja_template = re.sub(
                re.escape(context), "{{copy_context}}", jinja_template
            )
        else:
            _context = re.sub("{{\s*", r"{{\\s*", context)
            _context = re.sub("\s*}}", r"\\s*}}", _context)

            if dataset_name == "dbpedia_14":
                _context = re.sub("\s*\|\s*", r"\\s*\|\\s*", _context)
            if len(re.findall(_context, jinja_template)) > 0:
                jinja_template = re.sub(_context, "{{copy_context}}", jinja_template)
            else:
                continue
        assert "{{copy_context}}" != jinja_template.strip()
        jinja_template = re.sub("\|\|\|", "\n[pipe]\n", jinja_template)
        t.jinja = f"{context}|||" + jinja_template

        assert "{{copy_context}}" in jinja_template
        reverse_filtered_templates.append(t)
        filtered_templates.append(copy.deepcopy(templates[i]))

    return reverse_filtered_templates, filtered_templates


def get_task_type_dataset(
    dataset_name: str,
    dataset_config_name: Union[None, str],
    templates: list,
    task_type_df: pd.DataFrame,
):
    if dataset_config_name:
        dataset_df = task_type_df[
            (task_type_df["dataset_name"] == dataset_name)
            & (task_type_df["dataset_config_name"] == dataset_config_name)
        ]
    else:
        dataset_df = task_type_df[(task_type_df["dataset_name"] == dataset_name)]

    template_names = [template.name for template in templates]
    new_templates = []
    task_type = []
    for i, data in dataset_df.iterrows():
        if data["template_name"] not in template_names:
            print(
                data["template_name"],
                "not in template names",
                dataset_name,
                dataset_config_name,
            )
            continue

        task_type.append(data["task_type"])

        new_templates.append(templates[template_names.index(data["template_name"])])

    return task_type, new_templates


def create_task_generation_dataset(
    dataset_name, dataset_config_name, context, task_type_df, args
):
    task_generation_dataset = None
    # get all the templates from promptsource for the dataset
    reverse_templates, filtered_templates = get_filtered_template(
        dataset_name,
        dataset_config_name,
        context,
    )

    if len(reverse_templates) == 0:
        print(dataset_name, dataset_config_name, context, "has no templates")
        return None

    dataset = load_dataset(dataset_name, dataset_config_name)
    split = "train_r3" if dataset_name == "anli" else "train"

    # sample the max samples
    dataset_length = min(len(dataset[split]), args.max_examples)
    dataset_indices = random.sample(range(len(dataset[split])), dataset_length)
    dataset[split] = dataset[split].select(dataset_indices)

    column_names = dataset[split].column_names
    task_types, reverse_templates = get_task_type_dataset(
        dataset_name, dataset_config_name, reverse_templates, task_type_df
    )

    def preprocess_function(examples):
        bs = len(examples[column_names[0]])
        context_texts = []
        tasktype_texts = []
        task_input_texts = []
        task_target_texts = []

        for i in range(bs):
            ex = {k: examples[k][i] for k in column_names}
            random_index = random.randint(0, len(reverse_templates) - 1)
            random_template = reverse_templates[random_index]
            ex["copy_context"] = "{{context}}"

            input, target = random_template.apply(ex)

            # post process the task
            task = target.split("\n[pipe]\n")
            if len(task) == 1:
                task_input = task[0]
                task_output = ""
            else:
                task_input = task[0]
                task_output = task[1]
            task_input_texts.append(task_input.strip())
            task_target_texts.append(task_output.strip())

            tasktype_texts.append(task_types[random_index].strip())
            context_texts.append(input.strip())

        return {
            "context": context_texts,
            "task_input": task_input_texts,
            "task_output": task_target_texts,
            "dataset": [dataset_name] * bs,
            "dataset_config": [dataset_config_name] * bs,
            "task_type": tasktype_texts,
        }

    task_generation_dataset = dataset[split].map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        num_proc=12,
    )

    return task_generation_dataset


def create_training_pairs(dataset):
    def preprocess(examples):
        bs = len(examples["context"])
        inputs = []
        targets = []
        for i in range(bs):
            input_text = "<|tasktype|>\n" + examples["task_type"][i].strip()
            input_text += (
                "\n<|context|>\n" + examples["context"][i].strip() + "\n<|task|>\n"
            )
            target_text = (
                examples["task_input"][i].strip()
                + "\n<|pipe|>\n"
                + examples["task_output"][i].strip()
            )
            inputs.append(input_text)
            targets.append(target_text)

        return {
            "input": inputs,
            "output": targets,
        }

    return dataset.map(
        preprocess,
        batched=True,
        num_proc=24,
    )


def create_tasktype_bonito_dataset(args):
    datasets = {}
    task_type_df = pd.read_csv("ctga/bonito_prompts_task_types.csv")

    task_type_df.replace({np.nan: None}, inplace=True)

    for (dataset_name, dataset_config_name, context) in dataset_context_keys:

        _dataset = create_task_generation_dataset(
            dataset_name, dataset_config_name, context, task_type_df, args
        )

        if _dataset is not None:
            _dataset = _dataset.filter(lambda x: len(x["task_output"]) > 0)
            datasets[f"{dataset_name}-{dataset_config_name}"] = _dataset

    train_dataset = concatenate_datasets([d for n, d in datasets.items()])

    # create the final dataset
    train_dataset = create_training_pairs(train_dataset)

    all_dataset = DatasetDict({"train": train_dataset})

    # save data
    all_dataset.save_to_disk(os.path.join(args.output_dir, "text"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/bonito",
        help="The output directory where the dataset will be written.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=100000,
        help="The maximum number of examples to include from each dataset.",
    )
    args = parser.parse_args()
    create_tasktype_bonito_dataset(args)
    print("Done!")
