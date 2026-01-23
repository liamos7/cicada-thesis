import os
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from tensorboardX import SummaryWriter

from fastad.utils import CreateFolder, IsReadableDir, IsValidFile, IntOrIntListAction
from fastad.models import get_teacher_model
from fastad.trainers import BaseTrainer
from fastad.loggers import BaseLogger
from fastad.datasets import get_loaders


def main(args) -> None:

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = get_teacher_model(args.model, args.dataset, args.load_pretrained_path).to(device)

    #if args.verbose:
        #print(f"Model architecture:\n{model}")
        #from torchsummary import torchsummary
        #torchsummary.summary(model, input_size=(1, 18, 14))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    trainer = BaseTrainer(
        n_epochs=args.epochs, val_interval=args.val_interval, save_interval=args.save_interval, device=device,
    )

    # f"logs/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    writer = SummaryWriter(logdir=args.output, filename_suffix=".log")
    logger = BaseLogger(writer)

    train_loader, val_loader = get_loaders(
        hold_out_classes=args.holdout_class, batch_size=args.batch_size, ds_name=args.dataset, n_max=None, root=args.data_root_path,
    )

    for batch, labels in train_loader:
        print(f"Sample train batch shape: {batch.shape}, labels.unique: {labels.unique()}")
        break

    for batch, labels in val_loader:
        print(f"Sample validation batch shape: {batch.shape}, labels.unique: {labels.unique()}")
        break

    d_dataloaders = {"training": train_loader, "validation": val_loader}

    model, train_result = trainer.train(
        model, optimizer, d_dataloaders, logger=logger, logdir=writer.file_writer.get_logdir(), clip_grad=True
    )

    print(train_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", "-ds",
        type=str,
        choices=["MNIST", "FMNIST", "CIFAR10", "CICADA"],
        default="MNIST",
        help="Chose the dataset to train on"
    )
    parser.add_argument(
        "--data-root-path",
        type=str,
        default="./data",
        help="root path to the datasets (use path on shared filesystem, e.g. /scratch/... for Slurm jobs)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["AE", "VAE", "NAE", "NAEWithEnergyTraining"],
        default="AE",
        help="Chose the teacher's architecture",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for all RNGs. Set to None to not seed any RNGs.",
    )
    parser.add_argument(
        "--batch-size", "-bs",
        type=int,
        default=128,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--output", "-o",
        action=CreateFolder,
        type=Path,
        default="output/",
        help="Path to directory where models and logs will be stored",
    )
    parser.add_argument(
        "--holdout-class", "-ho",
        action=IntOrIntListAction,
        default=0,
        help="Which class(es) to use as holdout (=outlier,anomaly). Single integer or comma-separated list of integers",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        help="Number of training epochs",
        default=100,
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        help="after how many training iterations to validate (higher for faster training)",
        default=10000,
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        help="after how many training iterations to save the model",
        default=10000,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    parser.add_argument(
        "--load-pretrained-path",
        type=str,
        help="Path to pretrained autoencoder weights to load",
        default=None,
    )
    main(parser.parse_args())
