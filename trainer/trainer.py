import numpy as np
import torch
from torch._C import set_flush_denormal
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torchvision
from PIL import Image
from os.path import join as pjoin
import imageio

def tensorSaveExr(x, path):
    x = x.detach().cpu().numpy()
    x = x.transpose([1, 2, 0])
    if x.shape[2] == 2:
        zeros = np.zeros((x.shape[0], x.shape[1],1))
        x = np.concatenate([x, zeros], axis=2)

    imageio.imwrite(path, x.astype(np.float32))

def Detonemap(x):
    return torch.exp(x) - 1

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config["trainer"]["log_step"]
        self.save_img_step = config["trainer"]["save_img_step"]

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        # for batch_idx, (data, target) in enumerate(self.data_loader):
        for batch_idx, [view_list, depth_list, flow_list, truth] in enumerate(self.data_loader):
            # data, target = data.to(self.device), target.to(self.device)
            for i, item in enumerate(view_list):
                view_list[i] = item.to(self.device)
            for i, item in enumerate(depth_list):
                depth_list[i] = item.to(self.device)
            for i, item in enumerate(flow_list):
                flow_list[i] = item.to(self.device)
            target = truth.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(view_list, depth_list, flow_list)

            # Save batch result
            # toPILImage = torchvision.transforms.ToPILImage()
            # toPILImage(output[0]).save(pjoin(self.config.img_dir, f'epoch_{epoch}_batch_{batch_idx}.pred.png'))
            # toPILImage(truth[0]).save(pjoin(self.config.img_dir, f'epoch_{epoch}_batch_{batch_idx}.gt.png'))
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
            self.logger.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item())
                    )
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(view_list[0].cpu(), nrow=8, normalize=True))

            if batch_idx % self.save_img_step == 0:
                tensorSaveExr(Detonemap(output[0]), pjoin(self.config.img_dir, f'epoch_{epoch}_batch_{batch_idx}.pred.exr'))
                tensorSaveExr(Detonemap(view_list[0][0]), pjoin(self.config.img_dir, f'epoch_{epoch}_batch_{batch_idx}.input0.exr'))
                tensorSaveExr(Detonemap(view_list[1][0]), pjoin(self.config.img_dir, f'epoch_{epoch}_batch_{batch_idx}.input1.exr'))
                tensorSaveExr(Detonemap(view_list[2][0]), pjoin(self.config.img_dir, f'epoch_{epoch}_batch_{batch_idx}.input2.exr'))
                tensorSaveExr(Detonemap(view_list[3][0]), pjoin(self.config.img_dir, f'epoch_{epoch}_batch_{batch_idx}.input3.exr'))
                tensorSaveExr(Detonemap(view_list[4][0]), pjoin(self.config.img_dir, f'epoch_{epoch}_batch_{batch_idx}.input4.exr'))
                tensorSaveExr(Detonemap(truth[0]), pjoin(self.config.img_dir, f'epoch_{epoch}_batch_{batch_idx}.gt.exr'))
                
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        for key, value in log.items():
            self.writer.add_scalar("avg_" + key, value)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, [view_list, depth_list, flow_list, truth] in enumerate(self.data_loader):
                for i, item in enumerate(view_list):
                    view_list[i] = item.to(self.device)
                for i, item in enumerate(depth_list):
                    depth_list[i] = item.to(self.device)
                for i, item in enumerate(flow_list):
                    flow_list[i] = item.to(self.device)
                target = truth.to(self.device)

                output = self.model(view_list, depth_list, flow_list)
                loss = self.criterion(output, target)
                
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(view_list[0].cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
