import os
import random
import re
from argparse import ArgumentParser
from glob import glob
from importlib import import_module
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import *
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, \
    AutoModelForSequenceClassification

from load_data import *
from loss import *


class YamlConfigManager:
    def __init__(self, config_file_path, config_name):
        super().__init__()
        self.values = EasyDict()
        if config_file_path:
            self.config_file_path = config_file_path
            self.config_name = config_name
            self.reload()

    def reload(self):
        self.clear()
        if self.config_file_path:
            with open(self.config_file_path, 'r') as f:
                self.values.update(yaml.safe_load(f)[self.config_name])

    def clear(self):
        self.values.clear()

    def update(self, yml_dict):
        for (k1, v1) in yml_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1

    def export(self, save_file_path):
        if save_file_path:
            with open(save_file_path, 'w') as f:
                yaml.dump(dict(self.values), f)


# 실험을 위한 random seed 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# 실험을 위한 모델 저장 파일명 자동 변경
def increment_output_dir(output_path, exist_ok=False):
    path = Path(output_path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(cfg):
    # hyperparameter define
    SEED = cfg.values.seed
    seed_everything(SEED)
    data_path = args.data_path
    MODEL_NAME = cfg.values.model_name
    log_interval = cfg.values.train_args.log_interval
    weight_decay = cfg.values.train_args.weight_decay
    tr_batch_size = cfg.values.train_args.train_batch_size
    val_batch_size = cfg.values.train_args.eval_batch_size
    max_seqlen = cfg.values.train_args.max_seqlen
    epochs = cfg.values.train_args.num_epochs
    loss_type = cfg.values.train_args.loss_fn
    lr_decay_step = 1  # stepLR parameter
    steplr_gamma = cfg.values.train_args.steplr_gamma
    opti = cfg.values.train_args.optimizer
    scheduler_type = cfg.values.train_args.scheduler_type
    label_smoothing_factor = cfg.values.train_args.label_smoothing_factor
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device is "{device}"')
    print('BACKBONE_MODEL_NAME:', MODEL_NAME)
    print(os.getcwd())

    # data load
    format = data_path.split('.')[-1]
    if format == 'xlsx':
        train_df = pd.read_excel(args.data_path, engine='openpyxl')[[args.text_col, args.label_col]].dropna()
    elif format == 'csv':
        train_df = pd.read_csv(data_path)[[args.text_col, args.label_col]].dropna()
    train_df = train_df.reset_index(drop=True)
    train_df[args.label_col] = train_df[args.label_col].astype(int)
    train_label = train_df[args.label_col].values
    num_label = train_df[args.label_col].nunique()
    train_dataset = train_df[args.text_col]

    ##################################################################################
    #                              huggingface config
    ##################################################################################

    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = num_label

    ##################################################################################
    #                                    Tokenizer
    ##################################################################################

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_label),
                                         y=train_label)
    weights = torch.tensor(class_weights, dtype=torch.float)
    weights = weights.to(device)

    ##################################################################################
    #                                       LOSS
    ##################################################################################
    if loss_type == 'custom':  # F1 + Cross_entropy
        criterion = CustomLoss()
    elif loss_type == 'labelsmooth':
        criterion = LabelSmoothingLoss(smoothing=label_smoothing_factor)
    elif loss_type == 'CEloss':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        criterion = FocalLoss()

    ##################################################################################
    #                                 Training Start
    ##################################################################################
    # Data Imbalance 완화를 위한 Stratified kfold 적용
    kfold = StratifiedKFold(n_splits=cfg.values.val_args.n_splits)
    save_dir = increment_output_dir(cfg.values.train_args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print('=' * 100)
    print("SAVE DIR : ", save_dir)
    print('=' * 100)
    for idx, splits in enumerate(kfold.split(train_dataset, train_label)):
        trind, valind = splits
        tr_label = train_label[trind]
        val_label = train_label[valind]
        print(len(trind), len(valind))
        print(f'-----fold_{idx}의 train/val의 클래스 비율-------')
        for i in range(model_config.num_labels):
            print(
                f'{i}의 개수 : {tr_label.tolist().count(i) / len(tr_label):4.2%} / {val_label.tolist().count(i) / len(val_label):4.2%}')
        tr_dataset = train_dataset.iloc[trind]
        val_dataset = train_dataset.iloc[valind]
        tokenized_train = tokenized_dataset(tr_dataset, tokenizer, max_seqlen)
        tokenized_dev = tokenized_dataset(val_dataset, tokenizer, max_seqlen)
        Huno_train_dataset = HunoDataset(tokenized_train, tr_label)
        Huno_dev_dataset = HunoDataset(tokenized_dev, val_label)

        train_loader = DataLoader(Huno_train_dataset, batch_size=tr_batch_size, shuffle=True)
        val_loader = DataLoader(Huno_dev_dataset, batch_size=val_batch_size, shuffle=False)

        ##################################################################################
        #                                      Model
        ##################################################################################
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                                   config=model_config,
                                                                   )

        model.to(device)
        model_dir = save_dir

        ##################################################################################
        #                              Optimizer and Scheduler
        ##################################################################################
        if scheduler_type == 'stepLr':
            opt_module = getattr(import_module("torch.optim"), opti)
            optimizer = opt_module(
                filter(lambda p: p.requires_grad,
                       model.parameters()),
                lr=cfg.values.train_args.lr,
                weight_decay=weight_decay
            )
            scheduler = StepLR(optimizer, lr_decay_step,
                               gamma=steplr_gamma)  # 794) #gamma : 20epoch => lr x 0.01

        elif scheduler_type == 'cycleLR':
            opt_module = getattr(import_module("torch.optim"), opti)
            optimizer = opt_module(
                filter(lambda p: p.requires_grad,
                       model.parameters()),
                lr=cfg.values.train_args.lr,  # 5e-6,
                weight_decay=weight_decay
            )
            scheduler = CyclicLR(optimizer,
                                 base_lr=0.000000001,
                                 max_lr=cfg.values.train_args.lr,
                                 step_size_up=1,
                                 step_size_down=4,
                                 mode='triangular',
                                 cycle_momentum=False)

        # Tensorboard
        # logger = SummaryWriter(log_dir=model_dir)

        best_val_acc = 0
        best_val_loss = np.inf

        ##################################################################################
        #                              Training Loop
        ##################################################################################
        for epoch in range(epochs):
            model.train()
            loss_value = 0
            matches = 0
            for idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                # token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids,
                                # token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                labels=labels)

                loss = criterion(outputs[1], labels)
                loss_value += loss.item()
                preds = torch.argmax(F.log_softmax(outputs[1], dim=1), dim=-1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                matches += (preds == labels).sum().item()
                if (idx + 1) % log_interval == 0:
                    train_loss = loss_value / log_interval
                    train_acc = matches / tr_batch_size / log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr:.3}"
                    )
                    # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    # logger.add_scalar("Train/accuracy", train_acc,
                    #                   epoch * len(train_loader) + idx)
                    # logger.add_scalar("Train/lr", current_lr, epoch * len(train_loader) + idx)
                    loss_value = 0
                    matches = 0
                # optimizer train_loader for문 밖이면 에폭마다 업데이트, 안에 있으면 스텝마다 업데ㅌ이트
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                print("Calculating validation results...")
                val_loss_items = []
                val_acc_items = []
                for idx, val_batch in enumerate(val_loader):
                    input_ids = val_batch['input_ids'].to(device)
                    # token_type_ids = val_batch['token_type_ids'].to(device)
                    attention_mask = val_batch['attention_mask'].to(device)
                    labels = val_batch['labels'].to(device)

                    outputs = model(input_ids,
                                    # token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)

                    preds = torch.argmax(F.log_softmax(outputs[1], dim=1), dim=-1)
                    loss_item = outputs[0].item()
                    correct = preds.eq(labels)
                    acc_item = correct.sum().item()

                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_label)
                best_val_loss = min(best_val_loss, val_loss)

                if val_acc > best_val_acc:
                    print(
                        f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.state_dict(), f"./{model_dir}/best.pt")
                    best_val_acc = val_acc
                torch.save(model.state_dict(), f"./{model_dir}/last.pt")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                # logger.add_scalar("Val/loss", val_loss, epoch)
                # logger.add_scalar("Val/accuracy", val_acc, epoch)
                print()
        with open(f"./{model_dir}/config.yaml", 'w') as file:
            documents = yaml.dump(cfg.values, file)

        if cfg.values.val_args.fold_break:
            break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./train_data/final_hate_comments.csv")
    parser.add_argument("--config_file_path", type=str, default='./config.yml')
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--config", type=str, default="bisok")
    args = parser.parse_args()

    cfg = YamlConfigManager(args.config_file_path, args.config)
    pprint(cfg.values)

    train(cfg)
