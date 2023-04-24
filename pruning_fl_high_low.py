import torch
import torch.cuda
from torch import nn
from torch.nn import functional as F
import argparse
import gc
import itertools
import numpy as np
import os
import sys
import time
import pickle
from copy import deepcopy

from tqdm import tqdm
import warnings

from datasets import get_dataset, get_global_dataset
import models
from models import all_models, needs_mask, initialize_mask
import copy
import random

def device_list(x):
    if x == 'cpu':
        return [x]
    return [int(y) for y in x.split(',')]

parser = argparse.ArgumentParser()

parser.add_argument('--eta', type=float, help='learning rate', default=0.01)
parser.add_argument('--epochs', type=int, help='number of local epochs', default=10)
parser.add_argument('--pruning-begin', type=int, default=9, help='first epoch number when we should readjust')
parser.add_argument('--pruning-interval', type=int, default=10, help='epochs between readjustments')
parser.add_argument('--rounds-between-readjustments', type=int, default=10, help='rounds between readjustments')
parser.add_argument('--remember-old', default=False, action='store_true', help="remember client's old weights when aggregating missing ones")
parser.add_argument('--sparsity-distribution', default='erk', choices=('uniform', 'er', 'erk'))
parser.add_argument('--final-sparsity', type=float, default=None, help='final sparsity to grow to, from 0 to 1. default is the same as --sparsity')
parser.add_argument('--batch-size', type=int, default=32, help='local client batch size')
parser.add_argument('--l2', default=0.001, type=float, help='L2 regularization strength')
parser.add_argument('--momentum', default=0.9, type=float, help='Local client SGD momentum parameter')
parser.add_argument('--cache-test-set', default=False, action='store_true', help='Load test sets into memory')
parser.add_argument('--cache-test-set-gpu', default=False, action='store_true', help='Load test sets into GPU memory')
parser.add_argument('--test-batches', default=0, type=int, help='Number of minibatches to test on, or 0 for all of them')
parser.add_argument('--eval-every', default=1, type=int, help='Evaluate on test set every N rounds')
parser.add_argument('--min-votes', default=0, type=int, help='Minimum votes required to keep a weight')
parser.add_argument('--no-eval', default=True, action='store_false', dest='eval')
parser.add_argument('--grasp', default=False, action='store_true')
parser.add_argument('--fp16', default=False, action='store_true', help='upload as fp16')
parser.add_argument('--global-device', default='0', type=str)
parser.add_argument('--global_eval', default=True, action='store_false')
parser.add_argument('--min-samples', type=int, default=0, help='minimum number of samples required to allow a client to participate')
parser.add_argument('--samples-per-client', type=int, default=20, help='samples to allocate to each client (per class, for lotteryfl, or per client, for iid)')
parser.add_argument('--prox', type=float, default=0, help='coefficient to proximal term (i.e. in FedProx)')
parser.add_argument('--rate-decay-method', default='cosine', choices=('constant', 'cosine'), help='annealing for readjustment ratio')
parser.add_argument('--rate-decay-end', default=None, type=int, help='round to end annealing')


parser.add_argument('--clients', type=int, help='number of clients per round', default=10)
parser.add_argument('--total-clients', type=int, help='split the dataset between this many clients. Ignored for EMNIST.', default=10)
parser.add_argument('--rounds', type=int, help='number of global rounds', default=200)

parser.add_argument('--dataset', type=str, choices=('mnist', 'emnist', 'cifar10', 'cifar100'),
                    default='cifar10', help='Dataset to use')
parser.add_argument('--distribution', type=str, choices=('dirichlet', 'lotteryfl', 'iid'), default='dirichlet',
                    help='how should the dataset be distributed?')
parser.add_argument('--beta', type=float, default=0.1, help='Beta parameter (unbalance rate) for Dirichlet distribution')

parser.add_argument('--sparsity', type=float, default=0.0, help='sparsity from 0 to 1')

parser.add_argument('--grad-agg', default=False, action='store_true')
parser.add_argument('--Gamma', default=1.0, type=float)
parser.add_argument('--tradeoff', default=0.8, type=float)

parser.add_argument('--need-readjust', default=False, action='store_true')
parser.add_argument('--readjustment-ratio', type=float, default=0.3, help='readjust this many of the weights each time')

parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--device', default='0', type=device_list, help='Device to use for compute. Use "cpu" to force CPU. Otherwise, separate with commas to allow multi-GPU.')
parser.add_argument('-o', '--outfile', default='output.csv', type=str)

parser.add_argument('--high-sparsity', type=float, default=0.3, help='sparsity from 0 to 1')
parser.add_argument('--low-sparsity', type=float, default=0.3, help='sparsity from 0 to 1')


args = parser.parse_args()

rng = np.random.default_rng(0)

import csv
with open(args.outfile,'w') as f:
    csv_write = csv.writer(f)
    csv_head = ["server-round",
                "global-accuracy",
                "compute-times",
                "download-cost",
                "upload-cost",
                "largest-flops",
                "avg-flops",
                "dataset",
                "eta",
                "clients",
                "total-clients",
                "rounds",
                "distribution",
                "beta",
                "high-sparsity",
                "low-sparsity",
                "grad-agg",
                "Gamma",
                "tradeoff",
                "need-readjust",
                "readjustment-ratio"
                ]
    csv_write.writerow(csv_head)

devices = [torch.device(x) for x in args.device]
global_device = torch.device(args.device[0])
args.pid = os.getpid()

if args.rate_decay_end is None:
    args.rate_decay_end = (3 * args.rounds) // 4

def nan_to_num(x, nan=0, posinf=0, neginf=0):
    x = x.clone()
    x[x != x] = nan
    x[x == -float('inf')] = neginf
    x[x == float('inf')] = posinf
    return x.clone()

def evaluate_global(global_model, global_test_data):
    with torch.no_grad():
        correct = 0.
        total = 0.

        global_model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(global_test_data):
                outputs = global_model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                correct += sum(labels == outputs)
                total += len(labels)

        return correct / total

def dprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)
# Fetch and cache the dataset
dprint('Fetching dataset...')
cache_devices = devices

loaders = get_dataset(args.dataset, clients=args.total_clients, mode=args.distribution,
                      beta=args.beta, batch_size=args.batch_size, devices=cache_devices,
                      min_samples=args.min_samples, samples=args.samples_per_client, rng=rng)

if args.global_eval:
    global_test_data = get_global_dataset(args.dataset, clients=args.total_clients, mode=args.distribution,
                      beta=args.beta, batch_size=args.batch_size, devices=cache_devices,
                      min_samples=args.min_samples, samples=args.samples_per_client)

class Client:

    def __init__(self, id, device, train_data, net=models.MNISTNet,
                 local_epochs=10, learning_rate=0.01, dataset='mnist'):
        '''Construct a new client.

        Parameters:
        id : object
            a unique identifier for this client. For EMNIST, this should be
            the actual client ID.
        train_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us training samples.
        test_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us test samples.
            (we will use this as the validation set.)
        local_epochs : int
            the number of local epochs to train for each round

        Returns: a new client.
        '''

        self.id = id

        self.train_data = train_data

        self.device = device
        self.net = net(device=self.device).to(self.device)
        initialize_mask(self.net)
        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.reset_optimizer()

        self.local_epochs = local_epochs
        self.curr_epoch = 0

        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.initial_global_params = None

        self.update = {}

    def reset_optimizer(self):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=args.momentum, weight_decay=args.l2)

    def reset_weights(self, *args, **kwargs):
        return self.net.reset_weights(*args, **kwargs)
    
    def reset_mask(self, *args, **kwargs):
        return self.net.reset_mask(*args, **kwargs)
    
    def apply_mask(self, *args, **kwargs):
        return self.net.apply_mask(*args, **kwargs)

    def sparsity(self, *args, **kwargs):
        return self.net.sparsity(*args, **kwargs)

    def train_size(self):
        return sum(x[0].shape[0] for x in self.train_data)


    def train(self, global_params=None, initial_global_params=None,
              readjustment_ratio=0.5, readjust=False, sparsity=args.sparsity):
        '''Train the client network for a single round.'''

        ul_cost = 0.0
        dl_cost = 0.0
        flops = 0.0

        if global_params:
            # this is a FedAvg-like algorithm, where we need to reset
            # the client's weights every round
            self.reset_mask(global_params)
            self.net.layer_prune(sparsity=sparsity, sparsity_distribution=args.sparsity_distribution)


            # Try to reset the optimizer state.
            self.reset_optimizer()

            # if mask_changed:
            dl_cost += self.net.mask_size # need to receive mask

            if not self.initial_global_params:
                self.initial_global_params = initial_global_params
                # no DL cost here: we assume that these are transmitted as a random seed
            else:
                # otherwise, there is a DL cost: we need to receive all parameters masked '1' and
                # all parameters that don't have a mask (e.g. biases in this case)
                dl_cost += (1-sparsity) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)

        self.old_model = deepcopy(self.net.state_dict())
        for epoch in range(self.local_epochs):

            self.net.train()
            self.net = self.net.to(self.device)

            for inputs, labels in self.train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                if args.prox > 0:
                    loss += args.prox / 2. * self.net.proximal_loss(global_params)
                loss.backward()

                self.optimizer.step()

                self.apply_mask()
            
            data_size = len(self.train_data) * self.train_data[0][0].shape[0]
            h_in = self.train_data[0][0].shape[2]
            w_in = self.train_data[0][0].shape[3]
            flops += self.net.calculate_flops(data_size, h_in, w_in, sparsity)

            if args.need_readjust and args.sparsity:
                if (self.curr_epoch - args.pruning_begin) % args.pruning_interval == 0 and readjust:
                    prune_sparsity = sparsity + (1 - sparsity) * readjustment_ratio
                    # recompute gradient if we used FedProx penalty
                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
                    self.criterion(outputs, labels).backward()

                    self.net.layer_prune(sparsity=prune_sparsity, sparsity_distribution=args.sparsity_distribution)
                    self.net.layer_grow(sparsity=sparsity, sparsity_distribution=args.sparsity_distribution)

                    self.optimizer.step()
                    self.apply_mask()

                    ul_cost += (1-sparsity) * self.net.mask_size # need to transmit mask

            self.curr_epoch += 1

        # we only need to transmit the masked weights and all biases
        ul_cost += (1-sparsity) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)

        dprint(global_params['conv1.weight_mask'][0, 0, 0], '->', self.net.state_dict()['conv1.weight_mask'][0, 0, 0])
        dprint(global_params['conv1.weight'][0, 0, 0], '->', self.net.state_dict()['conv1.weight'][0, 0, 0])

        self.new_model = deepcopy(self.net.state_dict())
        
        for name, param in self.new_model.items():
            if name.endswith('_mask'):
                continue
            self.update[name] = (self.new_model[name] - self.old_model[name]).to(device=global_device, copy=True)

        ret = dict(state=self.net.state_dict(), update=self.update, dl_cost=dl_cost, ul_cost=ul_cost, flops=flops)
        return ret


# initialize clients
dprint('Initializing clients...')
clients = {}
client_ids = []

for i, (client_id, client_loaders) in tqdm(enumerate(loaders.items())):
    cl = Client(client_id, *client_loaders, net=all_models[args.dataset],
                learning_rate=args.eta, local_epochs=args.epochs)
    clients[client_id] = cl
    client_ids.append(client_id)
    torch.cuda.empty_cache()

# with open(args.outfile,'w') as f:
#     csv_write = csv.writer(f)
#     csv_head = client_ids
#     csv_write.writerow(csv_head)

# initialize global model
global_model = all_models[args.dataset](device=global_device)
initialize_mask(global_model)
            
if args.sparsity:
    global_model.layer_prune(sparsity=args.sparsity, sparsity_distribution=args.sparsity_distribution)

initial_global_params = deepcopy(global_model.state_dict())

# we need to accumulate compute/DL/UL costs regardless of round number, resetting only
# when we actually report these numbers
compute_times = np.zeros(len(clients)) # time in seconds taken on client-side for round
download_cost = np.zeros(len(clients))
upload_cost = np.zeros(len(clients))
FLOPs = np.zeros(len(clients))
important_score = np.zeros(len(clients))
updates = {}
local_stat = {}

high_index = [1,5,9,13,15]
low_index = [2,3,6,12,19]

# for each round t = 1, 2, ... do
for server_round in tqdm(range(args.rounds)):

    if server_round and args.sparsity:
        global_model.layer_prune(sparsity=args.sparsity, sparsity_distribution=args.sparsity_distribution)

    # sample clients
    # client_indices = rng.choice(list(clients.keys()), size=args.clients, replace=False)
    # client_indices = client_ids
    client_indices = high_index + low_index

    global_params = global_model.state_dict()

    # for each client k \in S_t in parallel do
    total_score = 0.0
    client_id = 0

    agg_update = {}
    update_model = {}
    cl_mask = {}
    aggregated_masks = {}
    tmp = None
    for name, param in global_params.items():
        if name.endswith('_mask'):
            continue
        agg_update[name] = torch.zeros_like(param, dtype=torch.float, device=global_device)
        update_model[name] = torch.zeros_like(param, dtype=torch.float, device=global_device)
        if needs_mask(name):
            update_model[name + '_mask'] = torch.ones_like(param, device=global_device)
            cl_mask[name] = torch.zeros_like(param, device=global_device)
            aggregated_masks[name] = torch.zeros_like(param, device=global_device)
    tmp = param.data.flatten().clone().to(device=global_device)

    for client_id in client_indices:
        client = clients[client_id]
        i = client_ids.index(client_id)

        readjust = False
        readjustment_ratio = 0.0
        if args.need_readjust:
            if args.rate_decay_method == 'cosine':
                readjustment_ratio = global_model._decay(server_round, alpha=args.readjustment_ratio, t_end=args.rate_decay_end)
            else:
                readjustment_ratio = args.readjustment_ratio
            readjust = (server_round - 1) % args.rounds_between_readjustments == 0 and readjustment_ratio > 0.
            if readjust:
                dprint('readjusting', readjustment_ratio)

        round_sparsity = args.sparsity

        if client_id in high_index:
            round_sparsity = args.high_sparsity
        else:
            round_sparsity = args.low_sparsity

        # Local client training.
        t0 = time.process_time()

        # actually perform training
        train_result = client.train(global_params=global_params, initial_global_params=initial_global_params,
                                    readjustment_ratio=readjustment_ratio,
                                    readjust=readjust, sparsity=round_sparsity)

        t1 = time.process_time()
        compute_times[i] = t1 - t0

        local_stat[client_id] = train_result['state']
        updates[client_id] = train_result['update']
        download_cost[i] = train_result['dl_cost']
        upload_cost[i] = train_result['ul_cost']
        FLOPs[i] = train_result['flops']

        client.net.clear_gradients() # to save memory

        for name, param in local_stat[client_id].items():
            # tmp_flattened = param.data.flatten().to(device=global_device, copy=True)
            # norm_value = torch.norm(param.data.flatten()) + 1e-7
            if name.endswith('_mask'):
                name = name[:-5]
                cl_mask[name] = param.clone().to(device=global_device)
        
        score = client.train_size()
        if server_round and args.grad_agg:
            score = important_score[i]
        total_score += score

        for name, param in updates[client_id].items():
            updates[client_id][name] = args.Gamma * updates[client_id][name]
            if name in aggregated_masks:
                updates[client_id][name] = updates[client_id][name] * cl_mask[name]
                aggregated_masks[name].add_(score * cl_mask[name])
            agg_update[name].add_(score * updates[client_id][name])


    agg_flattened = torch.randn(0).to(device=global_device)
    for name, param in global_params.items():
        if name.endswith('_mask'):
            continue
        if name in aggregated_masks:
            agg_update[name] /= aggregated_masks[name]
            agg_update[name] = torch.nan_to_num(agg_update[name], nan=0.0, posinf=0.0, neginf=0.0)
        else:
            agg_update[name] /= total_score
        update_model[name] = param.clone().to(device=global_device)
        update_model[name].add_(agg_update[name])
        
        tmp = agg_update[name].data.flatten().clone().to(device=global_device)
        agg_flattened = torch.cat((agg_flattened, tmp), dim=0)

    global_model.load_state_dict(update_model)

    total_important_score = 0.0
    for client_id in client_indices:
        flattened = torch.randn(0).to(device=global_device)
        for name, param in updates[client_id].items():
            tmp = param.data.flatten().clone().to(device=global_device)
            flattened = torch.cat((flattened, tmp), dim=0)
        i = client_ids.index(client_id)
        psi = torch.cosine_similarity(flattened, agg_flattened, dim=0)
        important_score[i] = args.tradeoff * important_score[i] + (1 - args.tradeoff) * psi
        important_score[i] = np.maximum(important_score[i], 1e-3)
        total_important_score += important_score[i]
    
    important_score /= total_important_score

    torch.cuda.empty_cache()
    if server_round % args.eval_every == 0 and args.global_eval:
        global_accuracy = evaluate_global(global_model, global_test_data)

    largest_compute_times = np.max(compute_times)
    avg_download_cost = np.sum(download_cost) / len(client_indices)
    avg_upload_cost = np.sum(upload_cost) / len(client_indices)
    avg_flops = np.sum(FLOPs) / len(client_indices)
    largest_flops = np.max(FLOPs)

    # output_data = []
    # for client_id in client_ids:
    #     i = client_ids.index(client_id)
    #     output_data.append(important_score[i])
    # output_data.append(global_accuracy.item())
    if server_round % args.eval_every == 0 and args.eval:
        with open(args.outfile,'a+') as f:
            csv_write = csv.writer(f)
            data_row = [server_round,
                        global_accuracy.item(),
                        largest_compute_times,
                        avg_download_cost,
                        avg_upload_cost,
                        largest_flops,
                        avg_flops,
                        args.dataset,
                        args.eta,
                        len(client_indices),
                        args.total_clients,
                        args.rounds,
                        args.distribution,
                        args.beta,
                        args.high_sparsity,
                        args.low_sparsity,
                        args.grad_agg,
                        args.Gamma,
                        args.tradeoff,
                        args.need_readjust,
                        args.readjustment_ratio]
            csv_write.writerow(data_row)

        # if we didn't send initial global params to any clients in the first round, send them now.
        # (in the real world, this could be implemented as the transmission of
        # a random seed, so the time and place for this is not a concern to us)
        if server_round == 0:
            clients[client_id].initial_global_params = initial_global_params

    if server_round % args.eval_every == 0 and args.eval:
        # clear compute, UL, DL costs
        compute_times[:] = 0.0
        download_cost[:] = 0.0
        upload_cost[:] = 0.0
        FLOPs[:] = 0.0