#encoding=utf-8

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

from torch_geometric.loader import DataLoader

from model.cost_estimator import CostEstimator
from qaoa_dataset import CostEstimatorDataset
from arguments import args
from utils.logger import setup_logger
from utils.config import configs

logger = setup_logger(args.log_dir, name='Cost-Estimator')

def train(model, loader, args, board):
    model.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.train.cuda and torch.cuda.is_available():
        model.cuda()

    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    params = [
        {'params': model.parameters(), 'initial_lr': args.train.lr_lp, 'lr': args.train.lr_lp},
    ]
    optimizer = torch.optim.Adam(params, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.train.decay_step, gamma=0.5, last_epoch=-1)

    for i in range(args.train.n_epoch):
        if device == 'cuda':
            model.cuda()
        for prob_graph, ansatz_graph, ps, label, keys in loader:
            pred = model(ansatz_graph.to(device), prob_graph.to(device), ps.to(device)).squeeze()
            label = label.to(device)
            if args.model.task == 'reg':
                loss_mse = torch.mean((pred - label)**2)
                pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
                label_diff = label.unsqueeze(1) - label.unsqueeze(0)
                loss_rank = nn.functional.relu(0 - torch.sign(label_diff) * pred_diff)
                loss_rank = torch.sum(loss_rank) / (len(pred)*(len(pred)-1)/2)

                loss = args.train.coef_mse_loss*loss_mse + args.train.coef_rank_loss*loss_rank
            else:
                loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        board.add_scalar('loss', loss.item(), i+1)
        board.add_scalar('lr', optimizer.param_groups[0]['lr'], i+1)
        scheduler.step()

        if (i+1) % args.train.save_freq == 0:
            torch.save(model.cpu().state_dict(), os.path.join(args.log_dir, 'model_e{}.pth'.format(i+1)))

if __name__ == '__main__':
    args.phase = 'loss'
    torch.manual_seed(0)
    configs.load(args.config, recursive=True)
    if torch.cuda.is_available():
        if not args.cuda:
            logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

    logger.info('begin')
    logger.info(f'arguments = {configs}')
    args.log_dir = os.path.join(args.log_dir, '/'.join(args.config.split('/')[1:])[:-5])
    configs.log_dir = args.log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    shutil.copy(args.config, args.log_dir)

    model = CostEstimator(configs)

    names = os.listdir(args.log_dir)
    for name in names:
        if 'event' in name:
            os.remove(os.path.join(args.log_dir, name))
    board = SummaryWriter(log_dir=args.log_dir)

    train_datatset = CostEstimatorDataset(configs.dataset, phase='train')
    logger.info(f'#samples={len(train_datatset)}')
    train_loader = DataLoader(train_datatset, batch_size=configs.train.bs_lp, shuffle=True, num_workers=configs.train.n_worker, pin_memory=True, drop_last=False)

    logger.info('----------begin to train-------------')
    train(model, train_loader, configs, board)
