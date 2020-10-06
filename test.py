import importlib
import os
import time
import random
import math

from functools import wraps

import sys
import copy
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from pysnooper import snoop
from memory_profiler import profile

import torch
import torch.nn as nn
from torch import multiprocessing
from torch.distributed import all_gather, get_world_size, is_initialized
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.modules.utils import _pair

from utils.model_profiling import model_profiling
from utils.transforms import Lighting
from utils.distributed import init_dist, master_only, is_master
from utils.distributed import get_rank, get_world_size
from utils.distributed import dist_all_reduce_tensor
from utils.distributed import master_only_print as mprint
from utils.distributed import AllReduceDistributedDataParallel, allreduce_grads
from train import get_experiment_setting, run_one_epoch
from utils.config import FLAGS
from utils.meters import ScalarMeter, flush_scalar_meters

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        if is_master():
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            mprint('func:{!r} took: {:2.4f} sec'.format(f.__name__, te-ts))
        else:
            result = f(*args, **kw)
        return result
    return wrap


def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes)
    if getattr(FLAGS, 'distributed', False):
        gpu_id = init_dist()
        if getattr(FLAGS, 'distributed_all_reduce', False):
            model_wrapper = AllReduceDistributedDataParallel(model.cuda())
        else:
            model_wrapper = torch.nn.parallel.DistributedDataParallel(
                model.cuda(), [gpu_id], gpu_id)
    else:
        model_wrapper = torch.nn.DataParallel(model).cuda()
    return model, model_wrapper


def data_transforms():
    """get transform of dataset"""
    if FLAGS.data_transforms in [
            'imagenet1k_basic', 'imagenet1k_inception', 'imagenet1k_mobile']:
        if FLAGS.data_transforms == 'imagenet1k_inception':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_basic':
            if getattr(FLAGS, 'normalize', False):
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            else:
                mean = [0.0, 0.0, 0.0]
                std = [1.0, 1.0, 1.0]
            #crop_scale = 0.08
            #jitter_param = 0.4
            #lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_mobile':
            if getattr(FLAGS, 'normalize', False):
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            else:
                mean = [0.0, 0.0, 0.0]
                std = [1.0, 1.0, 1.0]
            #crop_scale = 0.25
            #jitter_param = 0.4
            #lighting_param = 0.1
        train_transforms = transforms.Compose([
            #transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.RandomResizedCrop(224),# scale=(crop_scale, 1.0)),
            #transforms.ColorJitter(
            #    brightness=jitter_param, contrast=jitter_param,
            #    saturation=jitter_param),
            #Lighting(lighting_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms
    elif FLAGS.data_transforms == 'cifar':
        if getattr(FLAGS, 'normalize', False):
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        else:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]
        train_transforms = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])
        test_transforms = val_transforms
    elif FLAGS.data_transforms == 'cinic':
        if getattr(FLAGS, 'normalize', False):
            mean = [0.4789, 0.4723, 0.4305]
            std = [0.2421, 0.2383, 0.2587]
        else:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]
        train_transforms = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])
        test_transforms = val_transforms
    else:
        try:
            transforms_lib = importlib.import_module(FLAGS.data_transforms)
            return transforms_lib.data_transforms()
        except ImportError:
            raise NotImplementedError(
                'Data transform {} is not yet implemented.'.format(
                    FLAGS.data_transforms))
    return train_transforms, val_transforms, test_transforms


def dataset(train_transforms, val_transforms, test_transforms):
    """get dataset for classification"""
    if FLAGS.dataset == 'imagenet1k':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
        else:
            train_set = None
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        test_set = None
    elif FLAGS.dataset == 'imagenet1k_val50k':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
            seed = getattr(FLAGS, 'random_seed', 0)
            random.seed(seed)
            val_size = 50000
            random.shuffle(train_set.samples)
            train_set.samples = train_set.samples[val_size:]
        else:
            train_set = None
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        test_set = None
    elif FLAGS.dataset == 'CINIC10':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
        else:
            train_set = None
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'valid'),
            transform=val_transforms)
        test_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'test'),
            transform=val_transforms)
    elif FLAGS.dataset == 'CIFAR10':
        if not FLAGS.test_only:
            train_set = datasets.CIFAR10(
                FLAGS.dataset_dir,
                transform = train_transforms,
                download=True)
        else:
            train_set = None
        val_set = datasets.CIFAR10(
            FLAGS.dataset_dir,
            train=False,
            transform = val_transforms,
            download=True)
        test_set = None
    elif FLAGS.dataset == 'CIFAR100':
        if not FLAGS.test_only:
            train_set = datasets.CIFAR100(
                FLAGS.dataset_dir,
                transform = train_transforms,
                download=True)
        else:
            train_set = None
        val_set = datasets.CIFAR100(
            FLAGS.dataset_dir,
            train=False,
            transform = val_transforms,
            download=True)
        test_set = None
    else:
        try:
            dataset_lib = importlib.import_module(FLAGS.dataset)
            return dataset_lib.dataset(
                train_transforms, val_transforms, test_transforms)
        except ImportError:
            raise NotImplementedError(
                'Dataset {} is not yet implemented.'.format(FLAGS.dataset))
    return train_set, val_set, test_set


def data_loader(train_set, val_set, test_set):
    """get data loader"""
    train_loader = None
    val_loader = None
    test_loader = None
    if getattr(FLAGS, 'batch_size', False):
        if getattr(FLAGS, 'batch_size_per_gpu', False):
            assert FLAGS.batch_size == (FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job)
        else:
            assert FLAGS.batch_size % FLAGS.num_gpus_per_job == 0
            FLAGS.batch_size_per_gpu = (FLAGS.batch_size // FLAGS.num_gpus_per_job)
    elif getattr(FLAGS, 'batch_size_per_gpu', False):
        FLAGS.batch_size = FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job
    else:
        raise ValueError('batch size (per gpu) is not defined')
    batch_size = int(FLAGS.batch_size / get_world_size())
    if FLAGS.data_loader in ['imagenet1k_basic','cifar', 'cinic']:
        if getattr(FLAGS, 'distributed', False):
            if FLAGS.test_only:
                train_sampler = None
            else:
                train_sampler = DistributedSampler(train_set)
            val_sampler = DistributedSampler(val_set)
        else:
            train_sampler = None
            val_sampler = None
        if not FLAGS.test_only:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                pin_memory=True,
                num_workers=FLAGS.data_loader_workers,
                drop_last=getattr(FLAGS, 'drop_last', False))
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            num_workers=FLAGS.data_loader_workers,
            drop_last=getattr(FLAGS, 'drop_last', False))
        test_loader = val_loader
    else:
        try:
            data_loader_lib = importlib.import_module(FLAGS.data_loader)
            return data_loader_lib.data_loader(train_set, val_set, test_set)
        except ImportError:
            raise NotImplementedError(
                'Data loader {} is not yet implemented.'.format(
                    FLAGS.data_loader))
    if train_loader is not None:
        FLAGS.data_size_train = len(train_loader.dataset)
    if val_loader is not None:
        FLAGS.data_size_val = len(val_loader.dataset)
    if test_loader is not None:
        FLAGS.data_size_test = len(test_loader.dataset)
    return train_loader, val_loader, test_loader


def lr_func(x, fun='cos'):
    if fun == 'cos':
        return math.cos( x * math.pi ) / 2 + 0.5
    if fun == 'exp':
        return math.exp( - x * 8 )


def get_lr_scheduler(optimizer, nBatch=None):
    """get learning rate"""
    if FLAGS.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=FLAGS.multistep_lr_milestones,
            gamma=FLAGS.multistep_lr_gamma)
    elif FLAGS.lr_scheduler == 'exp_decaying':
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            if i == 0:
                lr_dict[i] = 1
            elif i % getattr(FLAGS, 'exp_decaying_period', 1) == 0:
                lr_dict[i] = lr_dict[i-1] * FLAGS.exp_decaying_lr_gamma
            else:
                lr_dict[i] = lr_dict[i-1]
        lr_lambda = lambda epoch: lr_dict[epoch]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'exp_decaying_iter':
        FLAGS.num_iters = FLAGS.num_epochs * nBatch
        FLAGS.warmup_iters = FLAGS.warmup_epochs * nBatch
        lr_dict = {}
        for i in range(FLAGS.warmup_iters):
            bs_ratio = 256 / FLAGS.batch_size
            lr_dict[i] = (1 - bs_ratio) / FLAGS.warmup_iters * i + bs_ratio
        for i in range(FLAGS.warmup_iters, FLAGS.num_iters):
            lr_dict[i] = lr_func((i - FLAGS.warmup_iters) / (FLAGS.num_iters - FLAGS.warmup_iters), 'exp')
        lr_lambda = lambda itr: lr_dict[itr] # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'linear_decaying':
        num_epochs = FLAGS.num_epochs - FLAGS.warmup_epochs
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            lr_dict[i] = 1. - (i - FLAGS.warmup_epochs) / FLAGS.num_epochs
        lr_lambda = lambda epoch: lr_dict[epoch]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'cos_annealing':
        num_epochs = FLAGS.num_epochs - FLAGS.warmup_epochs
        lr_dict = {}
        for  i in range(FLAGS.num_epochs):
            lr_dict[i] = (1.0 + math.cos( (i - FLAGS.warmup_epochs) * math.pi / num_epochs)) / 2
        lr_lambda = lambda epoch: lr_dict[epoch] # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'cos_annealing_iter':
        FLAGS.num_iters = FLAGS.num_epochs * nBatch
        FLAGS.warmup_iters = FLAGS.warmup_epochs * nBatch
        lr_dict = {}
        for i in range(FLAGS.warmup_iters):
            bs_ratio = 256 / FLAGS.batch_size
            lr_dict[i] = (1 - bs_ratio) / FLAGS.warmup_iters * i + bs_ratio
        if getattr(FLAGS, 'warm_restart', False):
            T = 10
            T_iter = T * nBatch
            start_iter = FLAGS.warmup_iters
            while True:
                if start_iter >= FLAGS.num_iters:
                    break
                T_iter = min(T_iter, FLAGS.num_iters - start_iter)
                for i in range(start_iter, start_iter + T_iter):
                    if i >= FLAGS.num_iters:
                        break
                    lr_dict[i] = (1.0 + math.cos((i - start_iter) * math.pi / T_iter)) / 2
                start_iter += T_iter
                T_iter *= 2
        else:
            for i in range(FLAGS.warmup_iters, FLAGS.num_iters):
                lr_dict[i] = (1.0 + math.cos((i - FLAGS.warmup_iters) * math.pi / (FLAGS.num_iters - FLAGS.warmup_iters))) / 2
        lr_lambda = lambda itr: lr_dict[itr] # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    else:
        try:
            lr_scheduler_lib = importlib.import_module(FLAGS.lr_scheduler)
            return lr_scheduler_lib.get_lr_scheduler(optimizer)
        except ImportError:
            raise NotImplementedError(
                'Learning rate scheduler {} is not yet implemented.'.format(
                    FLAGS.lr_scheduler))
    return lr_scheduler


def get_optimizer(model):
    """get optimizer"""
    if FLAGS.optimizer == 'sgd':
        # all depthwise convolution (N, 1, x, x) has no weight decay
        # weight decay only on normal conv and fc
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:
                weight_decay = FLAGS.weight_decay
            elif len(ps) == 2:
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                    'nesterov': FLAGS.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=FLAGS.lr, alpha=FLAGS.optim_decay, eps=FLAGS.optim_eps, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum)
    else:
        try:
            optimizer_lib = importlib.import_module(FLAGS.optimizer)
            return optimizer_lib.get_optimizer(model)
        except ImportError:
            raise NotImplementedError(
                'Optimizer {} is not yet implemented.'.format(FLAGS.optimizer))
    return optimizer


def set_random_seed(seed=None):
    """set random seed"""
    if seed is None:
        seed = getattr(FLAGS, 'random_seed', 0)
    print('seed for random sampling: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@master_only
def get_meters(phase):
    """util function for meters"""
    def get_single_meter(phase, suffix=''):
        meters = {}
        meters['loss'] = ScalarMeter('{}_loss/{}'.format(phase, suffix))
        for k in FLAGS.topk:
            meters['top{}_error'.format(k)] = ScalarMeter(
                '{}_top{}_error/{}'.format(phase, k, suffix))
        return meters

    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    meters = get_single_meter(phase)
    if phase == 'val':
        meters['best_val'] = ScalarMeter('best_val')
    return meters


def profiling(model, use_cuda):
    """profiling on either gpu or cpu"""
    mprint('Start model profiling, use_cuda:{}.'.format(use_cuda))
    flops, params, bitops, bitops_max, bytesize, energy, latency = model_profiling(
        model, FLAGS.image_size, FLAGS.image_size,
        verbose=getattr(FLAGS, 'model_profiling_verbose', False))
    return flops, params


#@profile
#@snoop(depth=2)
@timing
def train_val_test():
    """train and val"""
    torch.backends.cudnn.benchmark = True
    # init distributed
    if getattr(FLAGS, 'distributed', False):
        init_dist()
    # seed
    if getattr(FLAGS, 'use_diff_seed', False):
        print('use diff seed is True')
        while not is_initialized():
            print('Waiting for initialization ...')
            time.sleep(5)
        print('Expected seed: {}'.format(getattr(FLAGS, 'random_seed', 0) + get_rank()))
        set_random_seed(getattr(FLAGS, 'random_seed', 0) + get_rank())
    else:
        set_random_seed()
    # experiment setting
    experiment_setting = get_experiment_setting()

    # model
    model, model_wrapper = get_model()
    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    if getattr(FLAGS, 'profiling_only', False):
        if 'gpu' in FLAGS.profiling:
            profiling(model, use_cuda=True)
        if 'cpu' in FLAGS.profiling:
            profiling(model, use_cuda=False)
        return

    # data
    train_transforms, val_transforms, test_transforms = data_transforms()
    train_set, val_set, test_set = dataset(
        train_transforms, val_transforms, test_transforms)
    train_loader, val_loader, test_loader = data_loader(
        train_set, val_set, test_set)

    log_dir = FLAGS.log_dir
    log_dir = os.path.join(log_dir, experiment_setting)

    checkpoint = torch.load(
        os.path.join(log_dir, 'best_model.pt'),
        map_location=lambda storage, loc: storage)
    model_wrapper.load_state_dict(checkpoint['model'])
    optimizer = get_optimizer(model_wrapper)


    mprint('Start testing.')
    test_meters = get_meters('test')
    with torch.no_grad():
        run_one_epoch(
            -1, test_loader,
            model_wrapper, criterion, optimizer,
            test_meters, phase='test', ema=ema)


def init_multiprocessing():
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass


def main():
    """train and eval model"""
    init_multiprocessing()
    train_val_test()


if __name__ == "__main__":
    main()
