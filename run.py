import os
import math
import json
import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from data import MultitaskDataset
from model import InterSent

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def argLoader():
    parser = ArgumentParser()
    parser.add_argument("--do_train", action="store_true", help="train the model")
    parser.add_argument("--do_test", action="store_true", help="test the model")
    
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--batch_size_per_gpu", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--data_dir", type=str, default="./data")    
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_train_per_task", type=int, default=100000000)

    parser.add_argument("--encoder", type=str, default="roberta-base")
    parser.add_argument("--decoder", type=str, default="facebook/bart-base")
    parser.add_argument("--mask_prob", type=float, default=0.0)
    parser.add_argument("--pooler_type", type=str, default="cls")
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--bn_size", type=int, default=384)
    parser.add_argument("--cl_loss_weight", type=float, default=1.0)
    parser.add_argument("--gen_loss_weight", type=float, default=0.1)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--skip_mlp", action="store_true")

    parser.add_argument("--max_length", type=int, default=64)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fast_learning_rate", type=float, default=1e-3)

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--val_per_epoch", type=int, default=10)

    parser.add_argument("--tasks", nargs="+", default=["para", "add", "diff", "extcomp", "abscomp"])

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args

if __name__ == "__main__":
    args = argLoader()
    
    InterSentDataset = MultitaskDataset(args)

    num_train_examples = sum([len(dataset.train_dataset) for dataset in InterSentDataset.datasets.values()])
    args.train_steps = num_train_examples // (args.n_gpus * args.batch_size_per_gpu) * args.max_epochs
    args.warmup_steps = int(args.train_steps * args.warmup_ratio)


    if args.checkpoint is not None:
        model = InterSent.load_from_checkpoint(args.checkpoint, args=args, strict=False)
    else:
        model = InterSent(args)

    if args.do_train:
        rouge_checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{step}-{val_rouge1:.4f}-{val_stsb:.4f}",
            monitor="val_rouge1",
            mode="max",
            save_top_k=1,
            save_weights_only=True,
        )

        trainer = pl.Trainer(
            default_root_dir=args.output_dir,
            accelerator=args.accelerator,
            devices=args.n_gpus,
            strategy=args.strategy,
            precision=16,
            max_epochs=args.max_epochs,
            callbacks=[rouge_checkpoint_callback],
            sync_batchnorm=True,
            val_check_interval=1.0 / args.val_per_epoch,
            replace_sampler_ddp=False,
        )

        trainer.fit(model=model, datamodule=InterSentDataset)

    if args.do_test:
        if not args.do_train:
            trainer = pl.Trainer(
                default_root_dir=args.output_dir,
                accelerator=args.accelerator,
                devices=args.n_gpus,
                strategy=args.strategy,
                precision=16,
                sync_batchnorm=True,
            )

        for task in args.tasks:
            trainer.test(model=model, datamodule=InterSentDataset.datasets[task])
