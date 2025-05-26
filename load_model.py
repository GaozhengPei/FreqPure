import yaml
import torch
import torchvision.models as models
from robustbench.utils import load_model as load_clf

from score_sde.losses import get_optimizer
from score_sde.models import utils as mutils
from score_sde.models.ema import ExponentialMovingAverage
from score_sde import sde_lib
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from svhn.diffusion import Model as SVHNDiffusion
from svhn.resnet import resnet, SVHNClf
from utils import dict2namespace, restore_checkpoint,load_wideresnet_70_16
from torchvision.models import ResNet50_Weights


def load_models(args, model_src, device):
    if args.dataset == 'cifar10':
        # Diffusion model
        with open('./diffusion_configs/cifar10.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = dict2namespace(config)
        diffusion = mutils.create_model(config)
        optimizer = get_optimizer(config, diffusion.parameters())
        ema = ExponentialMovingAverage(
            diffusion.parameters(), decay=config.model.ema_rate)
        state = dict(step=0, optimizer=optimizer, model=diffusion, ema=ema)
        restore_checkpoint(model_src, state, device)
        ema.copy_to(diffusion.parameters())
        diffusion.eval().to(device)
        # # Underlying classifier
        # clf = load_clf(model_name='Standard',
        #                dataset='cifar10',threat_model='L2').to(device).eval()
        # clf = None
        clf = load_wideresnet_70_16().to(device).eval()
    elif args.dataset == 'imagenet':
        with open('./diffusion_configs/imagenet.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = dict2namespace(config)
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(config.model))
        diffusion, _ = create_model_and_diffusion(**model_config)
        diffusion.load_state_dict(torch.load(model_src, map_location='cpu'))
        diffusion.eval().to(device)

        # Underlying classifier
        clf = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
        # clf = models.resnet50(pretrained=True).to(device).eval()
    elif args.dataset == 'svhn':
        with open('./svhn/config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = dict2namespace(config)
        diffusion = SVHNDiffusion(config)
        diffusion.to(device)

        state = torch.load(model_src, map_location=device)[0]
        for key in list(state.keys()):
            state[key[7:]] = state.pop(key)
        diffusion.load_state_dict(state)
        diffusion.eval()

        # Underlying classifier
        from path import svhn_clf_path
        clf_forward, params = resnet(28, 10, 10)
        state_dict = torch.load(svhn_clf_path)
        params_tensors = state_dict['params']
        for k, v in params.items():
            v.data.copy_(params_tensors[k])
        for k, v in params.items():
            v.data = v.data.to(device)
        clf = SVHNClf(clf_forward, params)
    return clf, diffusion
