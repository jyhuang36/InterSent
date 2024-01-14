import os
import json
import random
import torch
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets, load_from_disk

# modified from https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/data/data_collator.py
class DataCollatorForInterSent:
    def __init__(self, tokenizer, mask_prob):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob

    def __call__(self, batch):
        inputs_a = [data["input_a"] for data in batch]
        inputs_b = [data["input_b"] for data in batch]
        inputs_target = [data["input_target"] for data in batch]

        # build inputs
        inputs = inputs_a + inputs_b + inputs_target
        inputs = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True)(inputs)
        inputs = self.mask_tokens(inputs)


        # reshape to (batch_size, sent_num, seq_le)
        input_seq_len = inputs["input_ids"].size(1)
        inputs["input_ids"] = inputs["input_ids"].contiguous().view(-1, 3, input_seq_len)
        inputs["attention_mask"] = inputs["attention_mask"].contiguous().view(-1, 3, input_seq_len)

        label_seq_len = inputs["labels"].size(1)
        inputs["labels"] = inputs["labels"].view(-1, 3, label_seq_len)

        return inputs


    def mask_tokens(self, inputs):
        probability_matrix = torch.full(inputs["input_ids"].shape, self.mask_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs["input_ids"].tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs["input_ids"][masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs


class InterSentDataset(LightningDataModule):
    def __init__(self, args, task):
        super().__init__()
        self.task = task
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.encoder, cache_dir=args.cache_dir)
        self.train_collator = DataCollatorForInterSent(self.tokenizer, args.mask_prob)
        self.val_collator = DataCollatorForInterSent(self.tokenizer, 0.0)
        self.test_collator = self.val_collator

        if not os.path.isdir(os.path.join(args.data_dir, "%s_processed/train"%task)):
            if task == "para":
                train_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "paranmt/para-nmt-5m-processed.txt"), split="train[:-5000]", cache_dir=args.cache_dir)
                val_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "paranmt/para-nmt-5m-processed.txt"), split="train[-5000:]", cache_dir=args.cache_dir)
                self.train_dataset = train_dataset.map(self.tokenize_pair, num_proc=args.num_workers) 
                self.val_dataset = val_dataset.map(self.tokenize_pair, num_proc=args.num_workers)
                self.test_dataset = self.val_dataset

            elif task == "add":
                train_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "discofuse/discofuse-train-balanced.txt"), split="train", cache_dir=args.cache_dir)
                val_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "discofuse/discofuse-dev-balanced.txt"), split="train[:5000]", cache_dir=args.cache_dir)
                test_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "discofuse/discofuse-test-balanced.txt"), split="train", cache_dir=args.cache_dir)
                self.train_dataset = train_dataset.map(self.tokenize_triplet, num_proc=args.num_workers) 
                self.val_dataset = val_dataset.map(self.tokenize_triplet, num_proc=args.num_workers)
                self.test_dataset = test_dataset.map(self.tokenize_triplet, num_proc=args.num_workers)

            elif task == "diff":
                train_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "wikisplit/wikisplit-train.txt"), split="train[:]", cache_dir=args.cache_dir)
                val_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "wikisplit/wikisplit-valid.txt"), split="train[:5000]", cache_dir=args.cache_dir)
                test_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "wikisplit/wikisplit-test.txt"), split="train", cache_dir=args.cache_dir)
                self.train_dataset = train_dataset.map(self.tokenize_triplet, num_proc=args.num_workers) 
                self.val_dataset = val_dataset.map(self.tokenize_triplet, num_proc=args.num_workers)
                self.test_dataset = test_dataset.map(self.tokenize_triplet, num_proc=args.num_workers)
                
            elif task == "extcomp":
                train_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "google/sent-comp-train.txt"), split="train", cache_dir=args.cache_dir)
                val_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "google/sent-comp-test.txt"), split="train[-5000:]", cache_dir=args.cache_dir)
                test_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "google/sent-comp-test.txt"), split="train[:1000]", cache_dir=args.cache_dir)
                self.train_dataset = train_dataset.map(self.tokenize_pair, num_proc=args.num_workers) 
                self.val_dataset = val_dataset.map(self.tokenize_pair, num_proc=args.num_workers)
                self.test_dataset = test_dataset.map(self.tokenize_pair, num_proc=args.num_workers) 

            elif task == "abscomp":
                train_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "gigaword/gigaword_train.txt"), split="train", cache_dir=args.cache_dir)
                val_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "gigaword/gigaword_valid.txt"), split="train[:5000]", cache_dir=args.cache_dir)
                test_dataset = load_dataset("text", data_files=os.path.join(args.data_dir, "gigaword/gigaword_test.txt"), split="train", cache_dir=args.cache_dir)
                self.train_dataset = train_dataset.map(self.tokenize_pair, num_proc=args.num_workers) 
                self.val_dataset = val_dataset.map(self.tokenize_pair, num_proc=args.num_workers)
                self.test_dataset = test_dataset.map(self.tokenize_pair, num_proc=args.num_workers) 

            else:
                raise NotImplementedError("%s not supported", task)

            self.train_dataset.save_to_disk(os.path.join(args.data_dir, "%s_processed/train"%task))
            self.val_dataset.save_to_disk(os.path.join(args.data_dir, "%s_processed/val"%task))
            self.test_dataset.save_to_disk(os.path.join(args.data_dir, "%s_processed/test"%task))
        
        else:
            self.train_dataset = load_from_disk(os.path.join(args.data_dir, "%s_processed/train"%task))
            self.val_dataset = load_from_disk(os.path.join(args.data_dir, "%s_processed/val"%task))
            self.test_dataset = load_from_disk(os.path.join(args.data_dir, "%s_processed/test"%task))

        self.train_dataset = self.train_dataset.select(range(min(args.max_train_per_task, len(self.train_dataset))))

    def tokenize_single(self, example):
        text_a, text_target = example["text"], example["text"]

        # build inputs without masking
        input_a = self.tokenizer(text_a, truncation=True, max_length=self.args.max_length)
        input_target = self.tokenizer(text_target, truncation=True, max_length=self.args.max_length)

        input_a["labels"] = input_target["input_ids"]
        input_target["labels"] = input_target["input_ids"]

        return {"input_a": input_a, "input_b": input_a, "input_target": input_target}

    def tokenize_pair(self, example):
        text_a, text_target = example["text"].split("\t")

        # build inputs without masking
        input_a = self.tokenizer(text_a, truncation=True, max_length=self.args.max_length)
        input_target = self.tokenizer(text_target, truncation=True, max_length=self.args.max_length)

        input_a["labels"] = input_target["input_ids"]
        input_target["labels"] = input_target["input_ids"]

        return {"input_a": input_a, "input_b": input_a, "input_target": input_target}

    def tokenize_triplet(self, example):
        text_a, text_b, text_target= example["text"].split("\t")

        # build inputs without masking
        input_a = self.tokenizer(text_a, truncation=True, max_length=self.args.max_length)
        input_b = self.tokenizer(text_b, truncation=True, max_length=self.args.max_length)
        input_target = self.tokenizer(text_target, truncation=True, max_length=self.args.max_length)

        input_a["labels"] = input_target["input_ids"]
        input_b["labels"] = input_target["input_ids"]
        input_target["labels"] = input_target["input_ids"]

        return {"input_a": input_a, "input_b": input_b, "input_target": input_target}

    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset) if self.args.n_gpus == 1 else DistributedSampler(self.train_dataset) 
        return DataLoaderWithTaskname(
            task=self.task,
            dataloader=DataLoader(
                dataset=self.train_dataset,
                batch_size=self.args.batch_size_per_gpu,
                sampler=sampler,
                num_workers=self.args.num_workers,
                collate_fn=self.train_collator,
                pin_memory=True,
                drop_last=True,
            )
        )

    def val_dataloader(self):
        sampler = RandomSampler(self.val_dataset) if self.args.n_gpus == 1 else DistributedSampler(self.val_dataset) 
        return DataLoaderWithTaskname(
            task=self.task,
            dataloader=DataLoader(
                dataset=self.val_dataset,
                batch_size=self.args.batch_size_per_gpu,
                sampler=sampler,
                num_workers=self.args.num_workers,
                collate_fn=self.val_collator,
                pin_memory=True,
                drop_last=True,
            )
        )

    def test_dataloader(self):
        sampler = RandomSampler(self.test_dataset) if self.args.n_gpus == 1 else DistributedSampler(self.test_dataset) 
        return DataLoaderWithTaskname(
            task=self.task,
            dataloader=DataLoader(
                dataset=self.test_dataset,
                batch_size=self.args.batch_size_per_gpu,
                sampler=sampler,
                num_workers=self.args.num_workers,
                collate_fn=self.test_collator,
                pin_memory=True,
            )
        )


class MultitaskDataset(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.datasets = {}
        
        for task in args.tasks:
            self.datasets[task] = InterSentDataset(args, task=task)

    def train_dataloader(self):
        return MultitaskDataloader({task: dataset.train_dataloader() for task, dataset in self.datasets.items()})

    def val_dataloader(self):
        return MultitaskDataloader({task: dataset.val_dataloader() for task, dataset in self.datasets.items()})

    def test_dataloader(self):
        return MultitaskDataloader({task: dataset.test_dataloader() for task, dataset in self.datasets.items()})


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task` string.
    This prevents it from throwing an error
    """
    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """
    def __init__(self, task, dataloader):
        self.task = task
        self.dataloader = dataloader

        self.batch_size = dataloader.batch_size
        self.dataset = dataloader.dataset

    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            batch["task"] = StrIgnoreDevice(self.task)
            yield batch

class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task: len(dataloader) 
            for task, dataloader in self.dataloader_dict.items()
        }
        self.task_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) 
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task in enumerate(self.task_list):
            task_choice_list += [i] * self.num_batches_dict[task]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task: iter(dataloader) 
            for task, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task = self.task_list[task_choice]
            yield next(dataloader_iter_dict[task]) 
