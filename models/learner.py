from __future__ import absolute_import, division

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb

import base.factory as factory
import utils.pth_utils as pth_utils
from base.base_learner import BaseLearner
from models.pruning import weight_pruning, unit_pruning, accuracy


class Learner(BaseLearner):
    """
    Learner for basic model
    """
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        wandb.init(project=os.getenv('PROJECT'), dir=os.getenv('LOG'), config=self.config, reinit=True)

        # construct the model
        model = factory.create_model(self.config)

        if self.config.TRAIN.RESTORE_FILE:
            checkpoint = pth_utils.restore_model(self.config.TRAIN.RESTORE_FILE, self.config.TRAIN.RESTORE_STORAGE)
            model.load_state_dict(checkpoint['state_dict'])

        wandb.watch(model, log='all')

        kwargs = {'num_workers': 8, 'pin_memory': True} \
            if torch.cuda.is_available() and not pth_utils.is_debug_session() else {}

        dataset = torchvision.datasets.MNIST(os.getenv('DATASETS'), train=True, download=True,
                                             transform=self.config.TRAIN.DATASET.TRANSFORM)

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[len(dataset) - 100, 100])
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.config.TRAIN.BATCH_SIZE,
                                                   shuffle=True,
                                                   **kwargs)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.config.VAL.BATCH_SIZE,
                                                 shuffle=False,
                                                 **kwargs)

        device = self.config.GPU
        model = model.to(device)
        params_to_update = model.parameters()
        optimizer = optim.Adam(params_to_update, lr=self.config.TRAIN.LR)
        criterion = nn.CrossEntropyLoss()
        step = 0

        best_model = {
            'step': step,
            'state_dict': copy.deepcopy(model.state_dict()),
            'loss': np.inf
        }

        # set model to train mode
        model.train()

        for epoch in range(self.config.TRAIN.NUM_EPOCHS):

            wandb.log({"train/epoch": epoch}, step=step)

            for samples in train_loader:
                inputs = samples[0]
                batch_size = inputs.shape[0]
                in_dim = inputs.shape[2] * inputs.shape[3]
                inputs_vec = inputs.reshape((batch_size, in_dim))
                labels = samples[1]
                inputs_vec = inputs_vec.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs_vec)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                probability_outputs = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probability_outputs, dim=1)
                if step % self.config.TRAIN.SUMMARY_FREQ == 0:
                    # log scalars
                    wandb.log({"train/loss": loss}, step=step)

                    # plot random sample and predicted class
                    sample_idx = np.random.choice(batch_size)
                    predicted_caption = str(predicted_classes[sample_idx].item())
                    gt_caption = str(labels[sample_idx].item())
                    caption = 'Prediction: {}\nGround Truth: {}'.format(predicted_caption, gt_caption)
                    wandb.log({"train/samples": wandb.Image(inputs[sample_idx], caption=caption)}, step=step)

                if step % self.config.TRAIN.VAL_FREQ == 0:
                    model.eval()
                    val_loss, val_acc = self._validate(model, criterion, val_loader, step, device)
                    wandb.log({"val/loss": val_loss}, step=step)
                    wandb.log({"val/accuracy": val_acc}, step=step)

                    if val_loss < best_model['loss']:
                        best_model['state_dict'] = copy.deepcopy(model.state_dict())
                        best_model['loss'] = val_loss
                        best_model['step'] = step
                    model.train()

                if step % self.config.TRAIN.SAVE_MODEL_FREQ == 0:
                    checkpoint_model = {
                        'step': step,
                        'state_dict': copy.deepcopy(model.state_dict())
                    }
                    pth_utils.save_model(checkpoint_model, 'model')

                step += 1

        model_name = self.config.MODEL.lower()
        pth_utils.save_model(best_model, model_name, upload_to_wandb=True)

    def test(self):
        wandb.init(project=os.getenv('PROJECT'), dir=os.getenv('LOG'), config=self.config, reinit=True)

        # construct the model
        model = factory.create_model(self.config)
        if not self.config.TEST.RESTORE_FILE:
            exit('Restore path is not set')

        checkpoint = pth_utils.restore_model(self.config.TEST.RESTORE_FILE, self.config.TEST.RESTORE_STORAGE)
        model.load_state_dict(checkpoint['state_dict'])

        kwargs = {'num_workers': 8, 'pin_memory': True} \
            if torch.cuda.is_available() and not pth_utils.is_debug_session() else {}

        dataset = torchvision.datasets.MNIST(os.getenv('DATASETS'), train=False, download=True,
                                             transform=self.config.TEST.DATASET.TRANSFORM)

        test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.TEST.BATCH_SIZE, shuffle=False,
                                                  **kwargs)
        criterion = nn.CrossEntropyLoss()
        device = self.config.GPU
        model = model.to(device)
        num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total number of parameters is {}'.format(num_of_params))
        model.eval()

        for k in self.config.TEST.PRUNING_K:
            # weight pruning
            pruned_model, wp_zeroed_weights = weight_pruning(model, k)
            wp_loss, wp_acc = self._validate(pruned_model, criterion, test_loader, k, device)
            model.load_state_dict(checkpoint['state_dict'])

            # unit pruning
            pruned_model, up_zeroed_weights = unit_pruning(model, k)
            up_loss, up_acc = self._validate(pruned_model, criterion, test_loader, k, device)
            model.load_state_dict(checkpoint['state_dict'])

            wandb.log({"pruning/weight/loss": wp_loss,
                       "pruning/unit/loss": up_loss,
                       "pruning/weight/accuracy": wp_acc,
                       "pruning/unit/accuracy": up_acc,
                       'pruning/k': k,
                       'pruning/weight/zeroed_weights': wp_zeroed_weights,
                       'pruning/unit/zeroed_weights': up_zeroed_weights})

    def _validate(self, model, criterion, val_loader, step, device):
        loss_meter = pth_utils.AverageMeter()
        p = []
        gt = []
        for samples in val_loader:
            inputs = samples[0]
            batch_size = inputs.shape[0]
            in_dim = inputs.shape[2] * inputs.shape[3]
            inputs_vec = inputs.reshape((batch_size, in_dim))
            labels = samples[1]
            inputs_vec = inputs_vec.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs_vec)
            loss = criterion(outputs, labels)

            probability_outputs = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probability_outputs, dim=1)
            p.extend(predicted_classes.tolist())
            gt.extend(labels.tolist())
            loss_meter.update(loss.item(), batch_size)

        p = np.array(p, dtype=np.int)
        gt = np.array(gt, dtype=np.int)
        acc = accuracy(p, gt)

        # log scalars
        val_loss = loss_meter.avg
        return val_loss, acc

    def inference(self):
        pass


def main():
    import importlib
    import copy
    config_module = importlib.import_module('models.basic_config')
    config = copy.deepcopy(config_module.cfg)
    learner = factory.create_learner(config)
    learner.test()


if __name__ == "__main__":
    main()
