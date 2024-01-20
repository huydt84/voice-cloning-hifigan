# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', message='.*kernel_size exceeds volume extent.*')

import itertools
import os
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
# from dataset import CodeDataset, mel_spectrogram, get_dataset_filelist
from dataset_custom import get_dataset_list, CustomCodeDataset, mel_spectrogram
from vocoder import  Vocoder, init_vocoder
from models import MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from utils_train import plot_spectrogram, scan_checkpoint, load_checkpoint, \
    save_checkpoint, build_env, AttrDict
    
from typing import Callable, Dict, List, Optional, Tuple, Union

import wandb

# wandb_id = wandb.util.generate_id()
# print(wandb_id)
wandb.init(
    # set the wandb project where this run will be logged
    project="vc-hifigan", 
    # id="9wj6t3m1",
    resume="allow",
    
    # track hyperparameters and run metadata
    config={
      "steps": 200,
    }
)

torch.backends.cudnn.benchmark = True

def train(a, h):

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:0')

    # generator: Vocoder = load_model_for_train(load_vocoder_model, "vocoder_36langs.yaml", device, torch.float32)
    vocoder: Vocoder = init_vocoder(a.model_config).to(device)
    generator = vocoder.code_generator
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    os.makedirs(a.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        print(f"Generator checkpoint: {cp_g}")
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)
    
    training_list, validation_list = get_dataset_list(h.training_metadata)
    print(f"Number of training files: {len(training_list)}")
    print(f"Number of validation files: {len(validation_list)}")
    trainset = CustomCodeDataset(training_list, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                           h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, 
                           fmax_loss=h.fmax_for_loss, device=device)

    train_loader = DataLoader(trainset, num_workers=0, shuffle=False,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)

    validset = CustomCodeDataset(validation_list, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                            h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                            fmax_loss=h.fmax_for_loss, device=device)
    validation_loader = DataLoader(validset, num_workers=0, shuffle=False, sampler=None,
                                    batch_size=h.val_batch_size, pin_memory=True, drop_last=True)

    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    for epoch in range(max(0, last_epoch), a.training_epochs):
        start = time.time()
        print("Epoch: {}".format(epoch + 1))

        for i, batch in enumerate(train_loader):
            generator.train()
            mpd.train()
            msd.train()
            start_b = time.time()
            
            x, y, _, y_mel = batch
            y = y.to(device)
            y_mel = y_mel.to(device)
            y = y.unsqueeze(1)
            x["code"] = torch.LongTensor(x["code"])
            x["spkr"] = torch.Tensor(x["spkr"])
            x["lang"] = torch.Tensor(x["lang"]).unsqueeze(-1)
            x = {k: v.to(device) for k, v in x.items()}

            y_g_hat = generator(x)
            if h.get('f0_vq_params', None) or h.get('code_vq_params', None):
                y_g_hat, commit_losses, metrics = y_g_hat

            assert y_g_hat.shape == y.shape, f"Mismatch in vocoder output shape - {y_g_hat.shape} != {y.shape}"
            if h.get('f0_vq_params', None):
                f0_commit_loss = commit_losses[1][0]
                f0_metrics = metrics[1][0]
            if h.get('code_vq_params', None):
                code_commit_loss = commit_losses[0][0]
                code_metrics = metrics[0][0]

            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                          h.win_size, h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            if h.get('f0_vq_params', None):
                loss_gen_all += f0_commit_loss * h.get('lambda_commit', None)
            if h.get('code_vq_params', None):
                loss_gen_all += code_commit_loss * h.get('lambda_commit_code', None)

            loss_gen_all.backward()
            optim_g.step()


            # STDOUT logging
            if steps % a.stdout_interval == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                wandb.log({"loss_gen_all1": loss_gen_all, "mel_error1": mel_error})
                print(
                    'Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format(steps,
                                                                                                                loss_gen_all,
                                                                                                                mel_error,
                                                                                                                time.time() - start_b))

            # checkpointing
            if steps % a.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'generator': generator.state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                                    'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                                                    'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(),
                                                    'steps': steps, 'epoch': epoch})

            # Tensorboard summary logging
            if steps % a.summary_interval == 0 and steps != 0:
                sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                sw.add_scalar("training/mel_spec_error", mel_error, steps)
                if h.get('f0_vq_params', None):
                    sw.add_scalar("training/commit_error", f0_commit_loss, steps)
                    sw.add_scalar("training/used_curr", f0_metrics['used_curr'].item(), steps)
                    sw.add_scalar("training/entropy", f0_metrics['entropy'].item(), steps)
                    sw.add_scalar("training/usage", f0_metrics['usage'].item(), steps)
                if h.get('code_vq_params', None):
                    sw.add_scalar("training/code_commit_error", code_commit_loss, steps)
                    sw.add_scalar("training/code_used_curr", code_metrics['used_curr'].item(), steps)
                    sw.add_scalar("training/code_entropy", code_metrics['entropy'].item(), steps)
                    sw.add_scalar("training/code_usage", code_metrics['usage'].item(), steps)

            # Validation
            if steps % a.validation_interval == 0 and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                start_v = time.time()
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        x, y, _, y_mel = batch
                        y = y.to(device)
                        y_mel = y_mel.to(device)
                        y = y.unsqueeze(1)
                        
                        x["code"] = torch.LongTensor(x["code"])
                        x["spkr"] = torch.Tensor(x["spkr"])
                        x["lang"] = torch.Tensor(x["lang"]).unsqueeze(-1)
                        x = {k: v.to(device) for k, v in x.items()}

                        y_g_hat = generator(x)
                        if h.get('f0_vq_params', None) or h.get('code_vq_params', None):
                            y_g_hat, commit_losses, _ = y_g_hat

                        if h.get('f0_vq_params', None):
                            f0_commit_loss = commit_losses[1][0]
                            val_err_tot += f0_commit_loss * h.get('lambda_commit', None)

                        if h.get('code_vq_params', None):
                            code_commit_loss = commit_losses[0][0]
                            val_err_tot += code_commit_loss * h.get('lambda_commit_code', None)
                        y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
                        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                        h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                        if j <= 4:
                            if steps == 0:
                                sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(y_mel[0].cpu()), steps)

                            sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                            y_hat_spec = mel_spectrogram(y_g_hat[:1].squeeze(1), h.n_fft, h.num_mels,
                                                            h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                            sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                            plot_spectrogram(y_hat_spec[:1].squeeze(0).cpu().numpy()), steps)

                    val_err = val_err_tot / (j + 1)
                    sw.add_scalar("validation/mel_spec_error", val_err, steps)
                    if h.get('f0_vq_params', None):
                        sw.add_scalar("validation/commit_error", f0_commit_loss, steps)
                    if h.get('code_vq_params', None):
                        sw.add_scalar("validation/code_commit_error", code_commit_loss, steps)
                wandb.log({"valid_mel_error1": val_err})
                print(
                    'Steps : {:d}, Validation Mel-Spec Error : {:4.3f}, s/b : {:4.3f}'.format(steps,
                                                                                                      val_err,
                                                                                                      time.time() - start_v))
                generator.train()

            steps += 1
            if steps >= a.training_steps:
                break

        scheduler_g.step()
        scheduler_d.step()

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

    print('Finished training')

    wandb.finish()


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='spk_enc/vocoder/train_config.json')
    parser.add_argument('--model_config', default='spk_enc/vocoder/model_config.json')
    parser.add_argument('--training_epochs', default=2, type=int)
    parser.add_argument('--training_steps', default=40001, type=int)
    parser.add_argument('--stdout_interval', default=2, type=int)
    parser.add_argument('--checkpoint_interval', default=200, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=100, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)

    train(a, h)


if __name__ == '__main__':
    main()