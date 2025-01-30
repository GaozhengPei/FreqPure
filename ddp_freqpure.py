import os
import datetime
import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torchvision.utils as tvu

from attacks.pgd_eot import PGD
from attacks.pgd_eot_l2 import PGDL2
from attacks.pgd_eot_bpda import BPDA
from attacks.aa_eot_l2 import AutoAttackL2
from attacks.aa_eot_linf import AutoAttackLinf
from attacks.pgd_eot_bpda import BPDA
from load_data import load_dataset_by_name
from load_model import load_models
from purification import PurificationForward
from utils import copy_source
from path import *
from PIL import Image
import shutil

def save_img(idx,img,pred,y,name):
    
    begin = idx * args.batch_size
    for i in range(img.shape[0]):
        if pred.reshape(-1)[i] == y[i]:
            path = './pure_images/{}/correct'.format(name)
        else:
            path = './pure_images/{}/false'.format(name)
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path,'{}.png'.format(begin+i))
        Image.fromarray((img[i].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)).save(save_path)
    

def save(idx,img,name):
    
    begin = idx * args.batch_size
    path = './{}'.format(name)
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(img.shape[0]):
        save_path = os.path.join(path,'{}.png'.format(begin+i))
        Image.fromarray((img[i].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)).save(save_path)

def get_diffusion_params(max_timesteps, num_denoising_steps):
    max_timestep_list = [int(i) for i in max_timesteps.split(',')]
    num_denoising_steps_list = [int(i) for i in num_denoising_steps.split(',')]
    assert len(max_timestep_list) == len(num_denoising_steps_list)

    diffusion_steps = []
    for i in range(len(max_timestep_list)):
        diffusion_steps.append([i - 1 for i in range(max_timestep_list[i] // num_denoising_steps_list[i],
                               max_timestep_list[i] + 1, max_timestep_list[i] // num_denoising_steps_list[i])])
        max_timestep_list[i] = max_timestep_list[i] - 1

    return max_timestep_list, diffusion_steps


def predict(x, args, defense_forward, num_classes):
    ensemble = torch.zeros(x.shape[0], num_classes).to(x.device)
    for _ in range(args.num_ensemble_runs):
        _x = x.clone()

        pure_images,logits = defense_forward.get_img_logits(_x)
        pred = logits.max(1, keepdim=True)[1]
        for idx in range(x.shape[0]):
            ensemble[idx, pred[idx]] += 1
    pred = ensemble.max(1, keepdim=True)[1]
    
    return pure_images,pred


def test(rank, world_size, args):
    print('rank {} | world_size {} started'.format(rank, world_size))

    model_src = diffusion_model_path[args.dataset]
    is_imagenet = True if args.dataset == 'imagenet' else False
    dataset_root = imagenet_path if is_imagenet else './dataset'
    num_classes = 1000 if is_imagenet else 10

    # Device
    device = torch.device('cuda:{}'.format(rank))

    # Load dataset
    assert 512 % args.batch_size == 0
    testset = load_dataset_by_name(args.dataset, dataset_root, 256)
    testsampler = torch.utils.data.distributed.DistributedSampler(testset,
                                                                num_replicas=world_size,
                                                                rank=rank)
    testLoader = torch.utils.data.DataLoader(testset,
                                            batch_size=args.batch_size,
                                            num_workers=0,
                                            pin_memory=True,
                                            sampler=testsampler,
                                            drop_last=False)


    correct_nat = torch.tensor([0]).to(device)
    correct_adv = torch.tensor([0]).to(device)
    total = torch.tensor([0]).to(device)

    for idx, (x, y) in enumerate(testLoader):
        
        # Load models
        clf, diffusion = load_models(args, model_src, device)

        # Set diffusion process for attack and defense
        attack_forward = PurificationForward(clf=clf, diffusion=diffusion,strength=args.strength, is_imagenet=is_imagenet,ddim_steps=args.attack_ddim_steps,
                                            amplitude_cut_range=args.amplitude_cut_range,phase_cut_range=args.phase_cut_range,delta=args.delta,device=device)
        defense_forward = PurificationForward(clf=clf, diffusion=diffusion,strength=args.strength, is_imagenet=is_imagenet,ddim_steps=args.defense_ddim_steps,
                                            amplitude_cut_range=args.amplitude_cut_range,phase_cut_range=args.phase_cut_range,delta=args.delta,device=device)

        # Set adversarial attack
        if args.dataset == 'cifar10':
            print('[Dataset] CIFAR-10')
            if args.attack_method == 'pgd':  # PGD Linf
                eps = 8./255.
                attack = PGD(attack_forward, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot)
                print('[Attack] PGD Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                    args.n_iter, eps, args.eot))
            elif args.attack_method == 'pgd_l2':  # PGD L2
                eps = 0.5
                attack = PGDL2(attack_forward, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot)
                print('[Attack] PGD L2 | attack_steps: {} | eps: {} | eot: {}'.format(
                    args.n_iter, eps, args.eot))
            elif args.attack_method == 'bpda':  # BPDA
                eps = 8./255.
                attack = BPDA(attack_forward, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot)
                print('[Attack] BPDA Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                    args.n_iter, eps, args.eot))
            if args.attack_method == 'aa':
                eps = 8./255.
                attack = AutoAttackLinf(attack_forward, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot)
                print('[Attack] CIFAR10 | AutoAttack Linf | attack_steps: {} | eps: {} | eot: {}'.format(
                    args.n_iter, eps, args.eot))
            if args.attack_method == 'aa_l2':
                eps = 8./255.
                attack = AutoAttackL2(attack_forward, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot)
                print('[Attack] CIFAR10 | AutoAttack L2 | attack_steps: {} | eps: {} | eot: {}'.format(
                    args.n_iter, eps, args.eot))
        elif args.dataset == 'imagenet':
            print('[Dataset] ImageNet')
            if args.attack_method == 'aa':
                eps = 4./255.
                attack = AutoAttackLinf(attack_forward, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot)
                print('[Attack] ImageNet | AutoAttack Linf | attack_steps: {} | eps: {} | eot: {}'.format(
                    args.n_iter, eps, args.eot))
            if args.attack_method == 'aa_l2':
                eps = 4./255.
                attack = AutoAttackL2(attack_forward, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot)
                print('[Attack] ImageNet | AutoAttack L2 | attack_steps: {} | eps: {} | eot: {}'.format(
                    args.n_iter, eps, args.eot))
        elif args.dataset == 'svhn':
            print('[Dataset] SVHN')
            eps = 8./255.
            attack = PGD(attack_forward, attack_steps=args.n_iter,
                        eps=eps, step_size=0.007, eot=args.eot)
            print('[Attack] PGD Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                args.n_iter, eps, args.eot))

        clf.eval()
        diffusion.eval()
        x = x.to(device)
        y = y.to(device)

        x_adv = attack(x, y)


        with torch.no_grad():

            pure_nat,pred_nat = predict(x, args, defense_forward, num_classes)
            correct_nat += pred_nat.eq(y.view_as(pred_nat)).sum().item()


            pure_adv,pred_adv = predict(x_adv, args, defense_forward, num_classes)
            correct_adv += pred_adv.eq(y.view_as(pred_adv)).sum().item()


        total += x.shape[0]

        print('rank {} | {} | num_samples: {} | acc_nat: {:.3f}% | acc_adv: {:.3f}%'.format(
            rank, idx, total.item(), (correct_nat / total *
                                    100).item(), (correct_adv / total * 100).item()
        ))


    dist.barrier()

    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_nat, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_adv, op=dist.ReduceOp.SUM)
    print('rank {} | num_samples: {} | acc_nat: {:.3f}% | acc_adv: {:.3f}%'.format(
        rank, total.item(), (correct_nat / total *
                            100).item(), (correct_adv / total * 100).item()
    ))





def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument("--use_cuda", default='True',
                        help="Whether use gpu or not")
    parser.add_argument("--use_wandb", action='store_true',
                        help="Whether use wandb or not")
    parser.add_argument("--wandb_project_name",
                        default='test', help="Wandb project name")
    parser.add_argument('--exp', type=str, default='test', help='Experiment name')
    parser.add_argument("--dataset", type=str, default='cifar10',
                        choices=['cifar10', 'imagenet', 'svhn'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--strength', type=float, default=0.1)
    parser.add_argument('--amplitude_cut_range', type=int, default=5)
    parser.add_argument('--phase_cut_range', type=int, default=2)
    parser.add_argument('--delta', type=float, default=0.3)
    parser.add_argument('--attack_ddim_steps', type=int, default=10)
    parser.add_argument('--defense_ddim_steps', type=int, default=200)
    # Attack
    parser.add_argument("--attack_method", type=str, default='pgd',
                        choices=['pgd', 'pgd_l2', 'bpda','aa','aa_l2'])
    parser.add_argument('--n_iter', type=int, default=200,
                        help='The nubmer of iterations for the attack generation')
    parser.add_argument('--eot', type=int, default=20,
                        help='The number of EOT samples for the attack')


    parser.add_argument('--num_ensemble_runs', type=int, default=10,
                        help='The number of ensemble runs for purification in defense')


    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args




def init_processes(fn, args):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend='nccl')
    fn(rank, world_size, args)



import sys
import os

class Tee:
    def __init__(self, *file_names):
        self.file_objects = [sys.stdout] 
        self.file_objects.extend([open(file_name, 'a') for file_name in file_names])

    def write(self, message):
        for file_object in self.file_objects:
            file_object.write(message)
            file_object.flush()  

    def flush(self):
        for file_object in self.file_objects:
            file_object.flush()



if __name__ == '__main__':
    log_file = './output_logs.txt'
    sys.stdout = Tee(log_file) 

    args = parse_args()
    print(args)  
    init_processes(test, args)