import torch
import wandb
import os
import sys


def is_debug_session():
    gettrace = getattr(sys, 'gettrace', None)
    debug_session = not ((gettrace is None) or (not gettrace()))
    return debug_session


def save_model(model_dict, name, upload_to_wandb=False):
    with torch.no_grad():
        model_name = '{}-{}.pth'.format(name, model_dict['step'])
        checkpoint_file = os.path.join(wandb.run.dir, model_name)
        torch.save(model_dict, checkpoint_file)
        if upload_to_wandb:
            wandb.save(model_name)


def restore_model(file, storage='local',  encoding='utf-8'):
    if storage == 'wandb':
        parts = file.split('/')
        wandb_path = '/'.join(parts[:-1])
        wandb_file = parts[-1]
        restore_file = wandb.restore(wandb_file, run_path=wandb_path)
        checkpoint = torch.load(restore_file.name, encoding=encoding)
    elif storage == 'local':  # local storage
        checkpoint = torch.load(file, encoding=encoding)
    else:
        print('Unknown storage type')
        checkpoint = None

    return checkpoint


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

