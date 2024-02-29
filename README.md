# Bonito Experiments

Bonito is an open-source model for generating task-specific synthetic instruction tuning datasets conditioned on unannotated text.

This repo contains code to reproduce the experiments from the Bonito paper.
For the Bonito package, see the [bonito](https://github.com/BatsResearch/bonito) repo. 

## Table of Contents
- [Bonito Experiments](#bonito-experiments)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Training](#training)
  - [Evaluation](#evaluation)
    - [Ranked Evaluation](#ranked-evaluation)
    - [SQuAD Evaluation](#squad-evaluation)
  - [Training the Bonito Model](#training-the-bonito-model)
    - [Generating CTGA-v1 Training Dataset](#generating-ctga-v1-training-dataset)
    - [Training](#training-1)
  - [Generating Instruction Tuning Datasets with Bonito](#generating-instruction-tuning-datasets-with-bonito)
  - [Credits](#credits)
  - [Citation](#citation)

## Installation
To install all the relevant packages, run the following:
```
conda create -n bonito-experiments python==3.9
conda activate bonito-experiments
pip3 install -r requirements.txt
```

## Training
To train models, run the following script:
```bash
deepspeed training/train_decoder.py --model_name_or_path mistralai/Mistral-7B-v0.1 --supervision_source bonito --dataset_name pubmed_qa --output_dir output/models/bonito_pubmed_qa_mistral
```
Options:
- `model_name_or_path`: The model to train. We consider `{mistralai/Mistral-7B-v0.1, meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-Instruct-v0.2}` in our experiments. You can train on any language model of your choice. Default is `mistralai/Mistral-7B-v0.1`.
- `supervision_source`: The source of supervision to train the model. This includes either synthetic instruction instruction dataset, or unnoatated texts, or general instruction tuning dataset. Your choices include `{bonito, dapt, mistral_instruct, zephyr_beta, p3}`. Default is `bonito`.
- `dataset_name`: The synthetic dataset. Your choices include `{pubmed_qa, privacy_qa, squadshifts_nyt, squadshifts_amazon, squadshifts_reddit, contract_nli, vitaminc}`. All the datasets are retrieved from [BatsResearch/bonito-experiment](https://huggingface.co/datasets/BatsResearch/bonito-experiment).
- `checkpoint_model_id_or_path` (Optional): This loads the LoRA adapter instruction tuned on P3. This is dependent on the `model_name_or_path`.
Use `BatsResearch/Mistral-7B-v0.1-P3` for `mistralai/Mistral-7B-v0.1` and `BatsResearch/Llama-2-7b-hf-P3` for `meta-llama/Llama-2-7b-hf` model.
You can also pass a local checkpoint. Default is `None`.

Notes:
1. If you are using a multi-gpu environment, ensure you adjust the `per_device_train_batch_size` and `gradient_accumulation_steps` to achieve an effective batch size of 16.
2. We train the model for 10,000 steps. If the dataset has fewer than 160,000 samples, then we train for 1 epoch.

## Evaluation
We evaluate the pretrained and fine-tuned models on prompted datasets.
We use ranked evaluation for `pubmed_qa`, `privacy_qa`, `contract_nli`, and `vitaminc` and SQuAD evaluation for `squadshifts_nyt`, `squadshifts_amazon`, and `squadshifts_reddit`.
All the evaluation datasets are uploaded to [BatsResearch/bonito-experiment-eval](https://huggingface.co/datasets/BatsResearch/bonito-experiment-eval).

### Ranked Evaluation

The following script evaluates the model on the target dataset:

```bash
deepspeed evaluation/evaluate_decoder.py --dataset_name pubmed_qa --model_name_or_path mistralai/Mistral-7B-v0.1 --checkpoint_model_id_or_path <checkpoint_path> --output_dir results/bonito-mistral-pubmed_qa --bf16
```

Options:
- `checkpoint_model_id_or_path`: path to the checkpoint directory or the huggingface model id. This is the path to the trained model. Default is `None`.
- `model_name_or_path`: The model to evaluate. We consider `{mistralai/Mistral-7B-v0.1, meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-Instruct-v0.2}` in our experiments. You can evaluate any language model of your choice. Default is `mistralai/Mistral-7B-v0.1`.
- `dataset_name`: the evaluation dataset. Your choices include `{pubmed_qa, privacy_qa, contract_nli, vitaminc}`. Default is `None`.
- `output_dir`: the directory to save the evaluation results. Default is `results`.

Additional options:
- `template_name`: runs evaluation for a specific template in the dataset. See the jinja templates for See the jinja templates for `pubmed_qa`, `privacy_qa`, `contract_nli`, and `vitaminc` in `templates` directory.  in `templates` directory. Default is `None`.


### SQuAD Evaluation

The following script merges the base model with the checkpoint adapter and evaluates the model on five templates from the SQuADShifts dataset:

```bash
python3 evaluation/merge_and_evaluate_squad.py --dataset_name squadshifts_nyt --model_name_or_path mistralai/Mistral-7B-v0.1 --checkpoint_model_id_or_path <checkpoint_path> --output_dir results/bonito-mistral-squadshifts_nyt
```


Options:
- `checkpoint_model_id_or_path`: path to the checkpoint directory or the huggingface model id. This is the path to the trained model. Default is `None`.
- `model_name_or_path`: The model to evaluate. We consider `{mistralai/Mistral-7B-v0.1, meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-Instruct-v0.2}` in our experiments. You can evaluate any language model of your choice. Default is `mistralai/Mistral-7B-v0.1`.
- `dataset_name`: the evaluation dataset. Your choices include `{squadshifts_nyt, squadshifts_amazon, squadshifts_reddit}`. Default is `None`.
- `output_dir`: the directory to save the evaluation results. Default is `results`.


Notes:
1. We use `SQuADShifts` templates from [promptsource](https://github.com/bigscience-workshop/promptsource).
2. The merging operation saves a new model in the `scratch` directory. Change `--scratch` path to save the model in a different directory. Additionally ensure you have enough space to save the model.


## Training the Bonito Model

### Generating CTGA-v1 Training Dataset
To generate the CTGA-v1 dataset, run the following script:
```bash
python3 ctga/task_type_bonito.py --output_dir output/dataset/ctga-v1
```

### Training
To train the Bonito model, run the following script:
```bash
deepspeed training/train_bonito.py --model_name_or_path mistralai/Mistral-7B-v0.1 --training_type="bonito_training" --dataset_name ctga-v1 --output_dir output/model/bonito_ctga-v1_mistral --max_steps 100000 --max_eval_samples 10000 --save_steps 10000 --save_total_limit 10
```

## Generating Instruction Tuning Datasets with Bonito
To generate instruction tuning datasets, run the following script:
```bash
python3 generation/generate_data.py --model_name_or_path BatsResearch/bonito-v1 --output_dir output/dataset/contract_nli --dataset_name contract_nli --task_type nli
```

Options:
- `model_name_or_path`: The model to generate the synthetic dataset. You can use  `BatsResearch/bonito-v1` in our experiments. You can generate datasets using any language model of your choice. Default is `BatsResearch/bonito-v1`.
- `output_dir`: the directory to save the generated dataset. Default is `output/dataset`.
- `dataset_name`: the name of the dataset. Your choices include `{pubmed_qa, privacy_qa, squadshifts_nyt, squadshifts_amazon, squadshifts_reddit, contract_nli, vitaminc}`. Default is `None`.
- `task_type`: the task type of the dataset. Your choices include `{exqa, ynqa,nli,mcqa, qg,qa,coref,paraphrase,paraphrase_id,sent_comp,sentiment,summarization,text_gen,topic_class,wsd,te}`. Default is `None`.


##  Credits

The training code is adapted from [Q-LoRA](https://github.com/artidoro/qlora). The evaluation code is adapted from [t-zero](https://github.com/bigscience-workshop/t-zero).


## Citation
If you use Bonito in your research, please cite the following paper:
```
@article{bonito:arxiv24,
  Author = {Nihal V. Nayak and Yiyang Nan and Avi Trost and Stephen H. Bach},
  Title = {Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation},
  Volume = {arXiv:2402.18334 [cs.CL]},
  Year = {2024}}
```

