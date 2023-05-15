import argparse
from time import time
import torch
import torchvision
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.model as module_test
from parse_config import ConfigParser
import os
import logging
from PIL import Image
from pathlib import Path

import imageio
import numpy as np

toPILImage = torchvision.transforms.ToPILImage()
# change the fileset_name to change data set
fileset_name = 'bunker_seq1'

def tensorSaveExr(x, path):
    x = x.detach().cpu().numpy()
    x = x.transpose([1, 2, 0])
    if x.shape[2] == 2:
        zeros = np.zeros((x.shape[0], x.shape[1],1))
        x = np.concatenate([x, zeros], axis=2)

    imageio.imwrite(path, x.astype(np.float32))

def Detonemap(x):
    return torch.exp(x) - 1

# logger_path = os.path.join(os.getcwd(), 'checkpoints', fileset_name, 'test_info.log')

def main(config):
    logger = config.get_logger('test')


    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        data_dir_list = ["/home/M2_Disk/Songyin/Data/Bunker/Train/Seq1"],
        cropped_size = None,
        cropped_num = 0,
        augmentation = False,
        batch_size = 1,
        shuffle = False,
        num_workers = 4
    )

    # build model architecture
    model = config.init_obj('arch', module_test)
    # logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))

    # checkpoint = torch.load(config.resume)
    # state_dict = checkpoint['state_dict']
    # if config['n_gpu'] > 1:
    #     model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(config.resume)["state_dict"])

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device=device)
    total_time = 0.0
    total_time_metric = torch.zeros(5) # feature extraction, feature zero upsampling, warping, reweighting, reconstruction
    with torch.no_grad():
        for index,  [view_list, depth_list, flow_list, truth] in enumerate(tqdm(data_loader)):
            for i, item in enumerate(view_list):
                view_list[i] = item.to(device)
            for i, item in enumerate(depth_list):
                depth_list[i] = item.to(device)
            for i, item in enumerate(flow_list):
                flow_list[i] = item.to(device)
            target = truth.to(device)

            start = time()
            output= model(view_list, depth_list, flow_list)
            total_time+=time() - start

            # time_metric = model.get_time_metric()
            # save sample images, or do something with output here
            # batch size must be 1 to generate a continuous video

            output_path = os.path.join(os.getcwd(),'output_pic_test', fileset_name,'output')
            truth_path = os.path.join(os.getcwd(),'output_pic_test', fileset_name,'ground_truth')
            if not os.path.exists(output_path):
                os.makedirs(output_path)    
            if not os.path.exists(truth_path):
                os.makedirs(truth_path)

            tensorSaveExr(Detonemap(output[0]), os.path.join(output_path, "{:04d}.exr".format(index)))
            tensorSaveExr(Detonemap(truth[0]), os.path.join(truth_path, "{:04d}.exr".format(index)))

            # computing loss, metrics on test set
            # loss = loss_fn(output, target)
            batch_size = output.shape[0]
            # total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
            # total_time_metric+=time_metric

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)
    
    total_time_metric/=n_samples
    logger.info(f"Feature_extraction_time: {total_time_metric[0]}")
    logger.info(f"Zero_upsampling_time: {total_time_metric[1]}")
    logger.info(f"backward_warp_time: {total_time_metric[2]}")
    logger.info(f"Feature_reweighting_time: {total_time_metric[3]}")
    logger.info(f"Reconstruction_time: {total_time_metric[4]}")
    logger.info(f"Total_time: {total_time/n_samples}")



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    config = ConfigParser.from_args(args)
    with torch.no_grad():
        main(config)
