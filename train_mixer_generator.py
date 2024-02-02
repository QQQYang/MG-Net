#encoding=utf-8

import torch
from torch.utils.tensorboard import SummaryWriter
import os
import pennylane as qml
import shutil

from torch_geometric.loader import DataLoader

from model.cost_estimator import CostEstimator
from model.mixer_generator import MixerGenerator
from qaoa_dataset import MixerGeneratorDataset
from arguments import args
from utils.logger import setup_logger
from utils.config import configs

logger = setup_logger(args.log_dir, name='Mixer-Generator')

def train_qaoa(model, args):
    opt_b = qml.AdamOptimizer(args.lr_qaoa)
    opt_c = qml.AdamOptimizer(args.lr_qaoa)
    loss_min = float('inf')
    losses = []
    param_bs, param_cs = [model.param_b.tolist()], [model.param_c.tolist()]
    for i in range(args.n_qaoa_epoch):
        # grad_fn = qml.grad(model)
        # grad =grad_fn(model.param_b, model.param_c)

        param_b, loss = opt_b.step_and_cost(lambda param_b: model(param_b, model.param_c), model.param_b)
        # model.param_b = model.param_b._value
        model.param_c = opt_c.step(lambda param_c: model(model.param_b, param_c), model.param_c)
        model.param_b = param_b.copy()
        loss_min = min(loss_min, loss)
        losses.append(loss.tolist())
        param_bs.append(param_b.tolist())
        param_cs.append(model.param_c.tolist())
        # print('Iteration:{}, loss={}'.format(i, loss))
    return losses, param_bs, param_cs

def train(model_arch, model_loss, loader, args, board):
    model_arch.train()
    model_loss.eval()
    device = "cuda" if torch.cuda.is_available() and args.train.cuda else "cpu"
    if args.train.cuda and torch.cuda.is_available():
        model_arch.cuda()
        model_loss.cuda()

    # optimizer
    params = [
        {'params': model_arch.parameters(), 'initial_lr': args.train.lr_mp, 'lr': args.train.lr_mp},
    ]
    optimizer = torch.optim.Adam(params, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.train.decay_step, gamma=0.5, last_epoch=-1)

    n_iter = 0
    for i in range(args.train.n_epoch):
        if device == 'cuda':
            model_arch.cuda()
        for prob_graph, ansatz_graph, ps, label, keys in loader:
            ansatz_graph = model_arch(prob_graph.to(device), ansatz_graph.to(device), ps.to(device))
            pred = model_loss(ansatz_graph, prob_graph.to(device), ps.to(device)).squeeze()
            loss = torch.mean(pred)
            # loss = torch.mean(pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_iter += 1
        logger.info('Epoch={}, loss={}, lr={}, p={}, group={}'.format(i+1, loss.item(), optimizer.param_groups[0]['lr'], torch.mean(ps.float()), ansatz_graph.edge_attr.max()))

        board.add_scalar('loss', loss.item(), i+1)
        board.add_scalar('lr', optimizer.param_groups[0]['lr'], i+1)
        scheduler.step()

        if (i+1) % args.train.save_freq == 0:
            torch.save(model_arch.cpu().state_dict(), os.path.join(args.log_dir, 'mixer_model_p{}_e{}.pth'.format(torch.mean(ps.float()).int(), i+1)))

if __name__ == '__main__':
    args.phase = 'arch'
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

    names = os.listdir(args.log_dir)
    for name in names:
        if 'event' in name:
            os.remove(os.path.join(args.log_dir, name))
    board = SummaryWriter(log_dir=args.log_dir)

    model_loss = CostEstimator(configs)
    model_loss.load_state_dict(torch.load(os.path.join(args.log_dir, 'model_e1000.pth'))) # model_e200.pth
    model_arch = MixerGenerator(configs)

    train_datatset = MixerGeneratorDataset(configs.dataset, phase='train')
    train_loader = DataLoader(train_datatset, batch_size=configs.train.bs_mp, shuffle=True, num_workers=configs.train.n_worker, pin_memory=True, drop_last=False)

    train(model_arch, model_loss, train_loader, configs, board)
