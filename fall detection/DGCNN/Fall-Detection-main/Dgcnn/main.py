#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import CustomDataset
from model import DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/models'):
        os.makedirs('checkpoints/' + args.exp_name + '/models')

# ...

# ...

def train(args, io):
    custom_dataset = CustomDataset(root_dir='/content/pytorch/data/anotation1/data_real', num_points=args.num_points)
    train_data, val_data, test_data, train_labels, val_labels, test_labels = custom_dataset.split_data()

    train_loader = DataLoader([(train_data[i], train_labels[i]) for i in range(len(train_data))], batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader([(val_data[i], val_labels[i]) for i in range(len(val_data))], batch_size=args.test_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader([(test_data[i], test_labels[i]) for i in range(len(test_data))], batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")     
    if args.model == 'dgcnn':
      model = DGCNN(args).to(device)
    else:
      raise Exception("Not implemented")

    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss

    best_test_acc = 0
    train_accuracies = []
    test_accuracies = []

    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
          # Inside your training loop, make sure the input data and labels are on the same device and data type as the model
          data = data.to(device, dtype=torch.float32)  # Change the data type to float32
          label = label.to(device, dtype=torch.long)

          data = data.permute(0, 2, 1)
        

          batch_size = data.size()[0]
          opt.zero_grad()
          logits = model(data)
          loss = criterion(logits, label)
          loss.backward()
          opt.step()
          preds = logits.max(dim=1)[1]
          count += batch_size
          train_loss += loss.item() * batch_size
          train_true.append(label.cpu().numpy())
          train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,train_loss*1.0/count,
                                                                                  metrics.accuracy_score(
                                                                                      train_true, train_pred),
                                                                                  metrics.balanced_accuracy_score(
                                                                                      train_true, train_pred))
        io.cprint(outstr)

        # ####################
        # # Test
        # ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data = data.to(device, dtype=torch.float32)  # Change the data type to float32
            label = label.to(device, dtype=torch.long)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)

        # Append training accuracy to the list
        train_accuracies.append(metrics.accuracy_score(train_true, train_pred))

        Append test accuracy to the list
        test_accuracies.append(test_acc)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='annotation1', metavar='N',
                        choices=['modelnet40', 'annotation1'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=512,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
