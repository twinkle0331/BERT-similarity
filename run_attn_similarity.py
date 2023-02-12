#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

import transformers
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import INN
import torch.nn as nn

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default='sst2',
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: str = field(
        default='google/multiberts-seed_0',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    example_num: int = field(
        default=25,
        metadata={
            "help": "The number of inference steps"
        },
    )
    seed_x: Optional[int] = field(
        default=15,
        metadata={
            "help": "one seed to utilize different bert pretrained model"
        }
    )
    seed_y: Optional[int] = field(
        default=6,
        metadata={
            "help": "the other seed to utilize different bert pretrained model"
        }
    )
    train: bool = field(default=False, metadata={"help": "Training mode or inference mode."})
    methods: str = field(
        default="CKA",
        metadata={
            "help": "The number of inference steps"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.xavier_normal_(m.bias.data.unsqueeze(1))


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model_x_path = os.path.join(model_args.model_path,f"seed_{model_args.seed_x}")
    config = BertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_x_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_x_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = BertForSequenceClassification.from_pretrained(
        model_x_path,
        from_tf=bool(".ckpt" in model_args.model_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if "test" not in raw_datasets and "test_matched" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset")
    predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
    if data_args.max_predict_samples is not None:
        predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    # Remove the redundant column index, however, this error exhibits variability in its occurrence across different random seed values.
    train_dataset = train_dataset.remove_columns("idx")
    eval_dataset = eval_dataset.remove_columns("idx")
    predict_dataset = predict_dataset.remove_columns("idx")

    train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator)
    val_dataloader = DataLoader(eval_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(predict_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator)

    seed_x = model_args.seed_x
    model.to(training_args.device)
    model.eval()

    seed_y = model_args.seed_y
    model_y_path = os.path.join(model_args.model_path, f"seed_{model_args.seed_y}")
    model_y = BertForSequenceClassification.from_pretrained(
        model_y_path,
        from_tf=bool(".ckpt" in model_x_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model_y.to(training_args.device)
    model_y.eval()

    def extract_hidden_states(model, train_dataloader, step_num, batch_size):
        """
            Extract the hidden states of the model for a given dataloader.

            Parameters:
            model (torch.nn.Module): A PyTorch model to extract the hidden states from.
            train_dataloader (torch.utils.data.DataLoader): The data loader for the training set to extract the hidden states from.
            step_num (int): The number of steps to extract the hidden states for.
            batch_size (int): The size of the batches to extract the hidden states from.

            Returns:
            hidden_states (list): A list of hidden states for each sample in the dataloader. The shape of each hidden state is (layer_num, step_num*batch_size, hidden_size).
        """
        model.to(training_args.device)
        hidden_states = []
        for iter in tqdm(range(step_num)):
            model.train()
            step, inputs = train_dataloader_iter.__next__()
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(training_args.device)
            inputs['labels'] = torch.zeros_like(inputs['labels'])
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states_tmp = torch.cat(outputs['hidden_states'], dim=0).reshape(13, batch_size, 128, 768).detach().mean(axis=2)
                hidden_states.append(hidden_states_tmp)

        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states

    def extract_representation(model, train_dataloader, step_num, batch_size):
        attn_prob = []
        example_num = 0
        for iter in tqdm(range(step_num)):
            model.train()
            step, inputs = train_dataloader_iter.__next__()
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(training_args.device)
            inputs['labels'] = torch.zeros_like(inputs['labels'])
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
            attn_prob_tmp = torch.cat(outputs.attentions, dim=0).reshape(12, 64, 12, 128, 128).detach().mean(axis=3)
            attn_prob.append(attn_prob_tmp)
        attn_prob = torch.cat(attn_prob, dim=1)
        return attn_prob

    def extract_rep_to_train_fit_model(model, inputs, batch_size):
        """
            Extract the hidden states of the model for a given batch.

            Parameters:
            model (nn.Module): A PyTorch model that will be used to extract hidden states from the inputs.
            inputs (Tensor): A tensor containing the input data.
            batch_size (int): The batch size to use when processing the inputs.

            Returns:
            hidden_states (list): A list of hidden states for each sample in the dataloader. The shape of each hidden state is (layer_num, batch_size, hidden_size).
        """
        attn_prob = []
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        attn_prob_tmp = torch.cat(outputs.attentions, dim=0).reshape(12, 64, 12, 128, 128).detach().mean(axis=3)
        attn_prob.append(attn_prob_tmp)
        attn_prob = torch.cat(attn_prob, dim=1)
        return attn_prob

    step_num = model_args.example_num
    methods = model_args.methods
    from utils.similarity import CKA, CCA

    def compute_similarity(features_x, features_y, methods, fit_model=None, weight=None):
        if methods == "SVCCA":
            cca = CCA()
            score = cca.compute_svcca(features_x.cpu().numpy(), features_y.cpu().numpy(), weight)
        elif methods == "PWCCA":
            cca = CCA()
            score = cca.compute_pwcca(features_x.cpu().numpy(), features_y.cpu().numpy(), weight)
        elif methods == "CKA":
            cka = CKA()
            score = cka.linear_cka(features_x.cpu().numpy(), features_y.cpu().numpy())
        elif methods == "INN":
            y_pre,_,_ = fit_model(features_x)
            mse_loss = torch.nn.MSELoss()
            score = mse_loss(y_pre, features_y)
        return score

    def compute_attention_similarity(X, Y, num_head, num_attention_heads, methods, weight=None, fit_model=None):  # 12,6400,12,128,128(64)
        '''
        Args:
            X num_layer num_sample num_attention_head seq_len hidden
            Y num_layer num_sample num_attention_head seq_len hidden
        Return:
            attention similarity score
        '''
        # X, Y = X.mean(axis=3), Y.mean(axis=3)  # 12,6400,12,128
        layer_dim, head_dim = 0, 1
        similarity = np.zeros((num_head, num_head))
        for i in range(num_head):
            for j in range(i + 1):
                i_layer_idx = i // num_attention_heads
                i_head_idx = i % num_attention_heads
                j_layer_idx = j // num_attention_heads
                j_head_idx = j % num_attention_heads
                if type(weight) == np.ndarray:
                    weight_item = weight[i][j]
                else:
                    weight_item = None
                selected_x = X.select(layer_dim, i_layer_idx).select(head_dim, i_head_idx)
                selected_y = Y.select(layer_dim, j_layer_idx).select(head_dim, j_head_idx)
                similarity[i][j] = compute_similarity(selected_x, selected_y, methods, fit_model, weight_item)
                logger.info(
                    f"Similarity of layer {i_layer_idx} head {i_head_idx} and layer {j_layer_idx} head{j_head_idx} is {similarity[i][j]}")
        for j in range(num_head):
            for i in range(j):
                similarity[i][j] = similarity[j][i]

        return similarity

    def compute_cca_weight(X, Y, num_head, num_attention_heads, methods, weight=None, fit_model=None):  # 12,6400,12,128,128(64)
        '''
        Args:
            X num_layer num_sample num_attention_head seq_len hidden
            Y num_layer num_sample num_attention_head seq_len hidden
        Return:
            attention similarity score
        '''
        # X, Y = X.mean(axis=3), Y.mean(axis=3)  # 12,6400,12,128
        layer_dim, head_dim = 0, 1
        similarity = [[[] for col in range(num_head)] for row in range(num_head)]
        for i in range(num_head):
            for j in range(i + 1):
                i_layer_idx = i // num_attention_heads
                i_head_idx = i % num_attention_heads
                j_layer_idx = j // num_attention_heads
                j_head_idx = j % num_attention_heads
                selected_x = X.select(layer_dim, i_layer_idx).select(head_dim, i_head_idx)
                selected_y = Y.select(layer_dim, j_layer_idx).select(head_dim, j_head_idx)
                similarity[i][j] = compute_similarity(selected_x, selected_y, methods, fit_model=fit_model)
                logger.info(
                    f"weight of layer {i_layer_idx} head {i_head_idx} and layer {j_layer_idx} head{j_head_idx} is {similarity[i][j]}")
        for j in range(num_head):
            for i in range(j):
                similarity[i][j] = similarity[j][i]

        return similarity

    output_dir = os.path.join(training_args.output_dir, f"methods_{methods}")
    os.makedirs(output_dir, exist_ok=True)

    if model_args.train:
        if methods =="INN":
            fit_model = INN.Sequential(INN.Nonlinear(768, 'RealNVP')).to(training_args.device)
            fit_model.apply(weights_init)
            fit_model.train()
            optimizer = optim.Adam(fit_model.parameters(), lr=1e-6)
            # An scheduler is optional, but can help in flows to get the last bpd improvement
            scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)

    train_dataloader_iter = enumerate(train_dataloader)
    example_num = 10
    num_head = 144
    num_attention_heads = 12

    if model_args.train:
        if methods == "INN":
            for iter in tqdm(range(3000)):
                try:
                    step, inputs = train_dataloader_iter.__next__()
                    optimizer.zero_grad()
                    hidden_states_x = extract_rep_to_train_fit_model(model, inputs, batch_size=training_args.per_device_train_batch_size)
                    hidden_states_y = extract_rep_to_train_fit_model(model_y, inputs, batch_size=training_args.per_device_train_batch_size)
                    num_layer = hidden_states_x.shape[0]
                    for i in range(num_layer):
                        selected_x = hidden_states_x.select(0, i)
                        selected_y = hidden_states_y.select(0, i)

                        y_pre, logp, logdet = fit_model(selected_x)
                        MSELoss = torch.nn.MSELoss()
                        loss = MSELoss(y_pre, selected_y)
                        logger.info(f'loss:{loss}')
                        loss.backward()
                        optimizer.step()
                except:
                    train_dataloader_iter = enumerate(train_dataloader)
                scheduler.step()
            torch.save(fit_model.state_dict(), output_dir + f"trans_model_{seed_x}_to_{seed_y}.pth")

        elif methods == "SVCCA" or methods == "PWCCA":
            atten_prob_x = extract_representation(model, train_dataloader, step_num,
                                                  batch_size=training_args.per_device_train_batch_size)
            atten_prob_y = extract_representation(model_y, train_dataloader, step_num,
                                                  batch_size=training_args.per_device_train_batch_size)
            fit_model = None
            weight = compute_cca_weight(atten_prob_x, atten_prob_y, num_head, num_attention_heads, methods, fit_model=fit_model)
            np.save(os.path.join(output_dir, f"trans_mat_{seed_x}_{seed_y}.npy"), weight)

    else:
        if methods == "INN":
            fit_model.load_state_dict(torch.load(output_dir + f"trans_model_{seed_x}_to_{seed_y}.pth"))
            fit_model.eval()
            weight = None
        elif methods == "SVCCA" or methods == "PWCCA":
            weight = np.load(os.path.join(output_dir, f"trans_mat_{seed_x}_{seed_y}.npy"),
                             allow_pickle=True)
            fit_model = None
        with torch.no_grad():
            # Default for evaluating on training set. Set train_dataloader to be test_dataloader if you want to evaluate on the test set.
            atten_prob_x = extract_representation(model, train_dataloader, step_num, batch_size=training_args.per_device_train_batch_size)
            atten_prob_y = extract_representation(model_y, train_dataloader, step_num, batch_size=training_args.per_device_train_batch_size)
            similarity = compute_attention_similarity(atten_prob_x, atten_prob_y, num_head, num_attention_heads, methods, weight=weight, fit_model=fit_model)
        mat_file = os.path.join(output_dir, f"sim_seed_{seed_x}_{seed_y}.npy")
        np.save(mat_file, similarity)
        mat = np.load(mat_file)
        plt.imshow(mat)
        plt.savefig(os.path.join(output_dir,f"sim_seed_{seed_x}_{seed_y}.png"))

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()