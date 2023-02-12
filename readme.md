# MultiBERTsConsistency: A Code for Evaluating the Consistency of Multiple Pre-Trained Models

This code provides an implementation for evaluating the consistency of multi-lingual pre-trained models, such as MultiBERT, RoBERTa, etc. The consistency of these models is evaluated using several methods, including Canonical Correlation Analysis (CCA), Singular Vector Canonical Correlation Analysis (SVCCA), and Projection-weighted Correlation Analysis (PWCCA), as well as linear regression and our proposed Invertible Neural Network (INN) method.

### Dependencies

We recommend using Anaconda for setting up the environment of experiments:

```python
git clone https://github.com/twinkle0331/MultiBERTsConsistency.git
cd MultiBERTsConsistency/
conda create -n consistency python=3.8
conda install pytorch==1.11.0 torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Code Structure

The code is organized into the following modules:

- `scripts/`: Scripts for running experiments. 
- `utils/`: This module contains functions for CCA,PWCCA,SVCCA.

## Fine-tuning models

```python
python scripts/run_finetune.py
```

## Sanity check

1. Training for bijective methods

```python
python run_sanity_check.py --per_device_train_batch_size 64 --model_path <model_path> --task <task_name> --max_seq_length 128 --methods <method_name> --seed_x {seed_x_num} --seed_y {seed_y_num} --train --output_dir <train_result_dir>
```

where `<model_path>` is the path of the fine-tuned model on task `<task_name>`(e.g.,mnli, sst2, mrpc) and methods `method_name` is choosen from `"SVCCA"`, `"PWCCA"` ,`"Linear" ` ,`"NonBijINN"` `"INN"`.

2. Inference for bijective methods

```
python run_sanity_check.py --per_device_eval_batch_size 64 --model_path <model_path> --task <task_name> --max_seq_length 128 --methods <method_name> --seed_x {seed_x_num} --seed_y {seed_y_num} --output_dir <train_result_dir>
```

## Hidden states

1. Training bijective function for hidden states

```python
python run_hidden_similarity.py --per_device_train_batch_size 64 --model_path <model_path> --task <task_name> --max_seq_length 128 --methods <method_name> --seed_x {seed_x_num} --seed_y {seed_y_num} --train --output_dir <train_result_dir>
```

2. Inference for bijective methods

```
python run_hidden_similarity.py --per_device_train_batch_size 64 --model_path <model_path> --task <task_name> --max_seq_length 128 --methods <method_name> --seed_x {seed_x_num} --seed_y {seed_y_num} --output_dir <train_result_dir>
```

## Attention weights

1. Training bijective function for hidden states

```python
python run_attn_similarity.py --per_device_train_batch_size 64 --model_path <model_path> --task <task_name> --max_seq_length 128 --methods <method_name> --seed_x {seed_x_num} --seed_y {seed_y_num} --train --output_dir <train_result_dir>
```

2. Inference for bijective methods

```
python run_attn_similarity.py --per_device_eval_batch_size 64 --model_path <model_path> --task <task_name> --max_seq_length 128 --methods <method_name> --seed_x {seed_x_num} --seed_y {seed_y_num} --output_dir <train_result_dir>
```

## Stiching experiments

```python
python run_stiching_glue.py --per_device_eval_batch_size 64 --model_path <model_path> --task <task_name> --max_seq_length 128 --methods <method_name> --output_dir <train_result_dir> --fit_model_path <train_result_dir> --cca_weight_path <train_result_dir>
```



### Evaluation Metrics

The code reports the L2 norm after aligning the sentence space on the training set (for reference) and test set (as the main result). The training and test samples are randomly chosen from the training and test datasets, and the sample size is set to 1600. For reference, the performance of a non-bijective neural network (Non-Bij. NN) is also reported, which should result in the smallest L2 norm but violates the Bijective Hypothesis.