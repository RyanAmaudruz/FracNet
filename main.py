import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import wandb

from wandb.fastai import WandbCallback
from functools import partial
from torch import save

from fastai.basic_train import Learner
from fastai.train import ShowGraph, SaveModelCallback
from fastai.data_block import DataBunch
from torch import optim

from dataset.fracnet_dataset import FracNetTrainDataset
from dataset import transforms as tsfm
from utils.metrics import dice, recall, precision, fbeta_score
from model.unet import UNet
from model.losses import MixLoss, DiceLoss
from utils import get_wandb_run_name

def main(args):
    train_image_dir = args.train_image_dir
    train_label_dir = args.train_label_dir
    val_image_dir = args.val_image_dir
    val_label_dir = args.val_label_dir

    lr_max = args.lr_max
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    optimizer = optim.SGD
    criterion = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)

    thresh = args.thresh
    recall_partial = partial(recall, thresh=thresh)
    precision_partial = partial(precision, thresh=thresh)
    fbeta_score_partial = partial(fbeta_score, thresh=thresh)

    # Accelerate training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Model configuration
    in_channels = args.in_channels
    out_channels = args.out_channels
    first_out_channels = args.first_out_channels
    model = UNet(in_channels, out_channels, first_out_channels)
    model = model.to(device)
    model_weight_filename = f'{str(model)}_batch-{batch_size}_epoch-{epochs}_lr-{lr_max}'
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model.cuda())

    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
    ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,
        transforms=transforms)
    dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,
        num_workers)
    ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,
        transforms=transforms)
    dl_val = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,
        num_workers)

    databunch = DataBunch(dl_train, dl_val,
        collate_fn=FracNetTrainDataset.collate_fn)

    # Creating the run name based on model name and parser args (ToDo)
    config = {}
    config['batch_size'] = batch_size
    config['epochs'] = epochs
    config['lr_max'] = lr_max
    config['optimizer'] = optimizer
    config['criterion'] = criterion
    config['thresh'] = thresh
    config['in_channels'] = in_channels
    config['out_channels'] = out_channels
    config['first_out_channels'] = first_out_channels

    wandb_run_name = get_wandb_run_name(
        model_name='fracnet',
        # Extra wandb filename parameters are now supported.
    )

    wandb.init(project='ai4med', entity='msc-ai',
               config=config, reinit=True, name=wandb_run_name,
               tags=['latest'])

    learn = Learner(
        databunch,
        model,
        opt_func=optimizer,
        loss_func=criterion,
        metrics=[dice, recall_partial, precision_partial, fbeta_score_partial],
        callback_fns=WandbCallback
    )

    learn.fit_one_cycle(
        epochs,
        lr_max,
        pct_start=0,
        div_factor=1000,
        callbacks=[
            ShowGraph(learn),
            SaveModelCallback(learn, monitor='dice', mode='max', 
                              name=os.path.join(model_weights_dir, model_weight_filename))
        ]
    )

if __name__ == "__main__":
    data_directory = os.path.join(os.getcwd(), "data")
    train_image_dir = os.path.join(data_directory, "train", "ribfrac-train-images")
    train_label_dir = os.path.join(data_directory, "train", "ribfrac-train-labels")
    val_image_dir = os.path.join(data_directory, "val", "ribfrac-val-images")
    val_label_dir = os.path.join(data_directory, "val", "ribfrac-val-labels")

    # Model weights save dir
    model_weights_dir = os.path.join(os.getcwd(), "weights")
    os.makedirs(model_weights_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers.")
    parser.add_argument("--lr_max", type=float, default=1e-1,
                        help="Maximum learning rate.")
    parser.add_argument("--thresh", type=float, default=0.1,
                        help="Threshold for metrics.")
    parser.add_argument("--in_channels", type=int, default=1,
                        help="Number of input channels.")
    parser.add_argument("--out_channels", type=int, default=1,
                        help="Number of output channels.")
    parser.add_argument("--first_out_channels", type=int, default=16,
                        help="Number of first output channels.")
    parser.add_argument("--train_image_dir",
                        help="The training image nii directory.", default=train_image_dir)
    parser.add_argument("--train_label_dir",
                        help="The training label nii directory.", default=train_label_dir)
    parser.add_argument("--val_image_dir",
                        help="The validation image nii directory.", default=val_image_dir)
    parser.add_argument("--val_label_dir",
                        help="The validation label nii directory.", default=val_label_dir)
    parser.add_argument("--save_model", default=True,
                        help="Whether to save the trained model.")
    args = parser.parse_args()

    print(args)
    main(args)
