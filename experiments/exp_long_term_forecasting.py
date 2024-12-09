from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm
import wandb

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if not self.args.cached_dataset:
            data_set, data_loader = data_provider(self.args, flag)
        elif not os.path.exists(os.path.join(self.args.root_path, 'data_set_' + flag + '.pkl')):
            data_set, data_loader = data_provider(self.args, flag)
            torch.save(data_set, os.path.join(self.args.root_path, 'data_set_' + flag + '.pkl'))
        else:
            data_set = torch.load(os.path.join(self.args.root_path, 'data_set_' + flag + '.pkl'))
            data_set, data_loader = data_provider(self.args, flag, data_set)
        
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_mae = []
        num_data = 0
        mean = vali_data.scaler.mean_[-1]
        scale = vali_data.scaler.scale_[-1]
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.long().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.long().to(self.device)
                mask_batch_y = torch.zeros_like(batch_y).to(self.device)
                
                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, mask_batch_y, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, mask_batch_y, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, mask_batch_y, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, mask_batch_y, batch_y_mark)
                
                outputs = torch.clamp_min(outputs, -mean/scale)
                batch_y = batch_y[:, :, -1:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                loss = criterion(pred, true)
                
                pred = pred * scale + mean
                true = true * scale + mean

                mae = (pred - true).abs().mean()

                total_loss.append(loss * batch_x.size(0))
                total_mae.append(mae * batch_x.size(0))
                num_data += batch_x.size(0)
                
        total_loss = np.sum(total_loss) / num_data
        total_mae = np.sum(total_mae) / num_data
        self.model.train()
        return total_loss, total_mae

    def train(self, setting):
        if self.args.wandb:
            wandb.init(project='Tbrain', name=setting, config=self.args)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        
        mean = train_data.scaler.mean_[-1]
        scale = train_data.scaler.scale_[-1]

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = CosineLRScheduler(
            optimizer=model_optim, 
            t_initial=self.args.train_epochs, 
            warmup_t=5, 
            warmup_lr_init=0.1 * self.args.learning_rate,
            warmup_prefix=True,
            lr_min=1e-8,
        )

        if self.args.use_amp:
            scaler = torch.amp.GradScaler('cuda')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            num_data = 0

            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.long().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.long().to(self.device)
                mask_batch_y = torch.zeros_like(batch_y).to(self.device)

                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, mask_batch_y, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, mask_batch_y, batch_y_mark)

                        outputs = torch.clamp_min(outputs, -mean/scale)
                        loss = criterion(outputs, batch_y[:, :, -1:])
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, mask_batch_y, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, mask_batch_y, batch_y_mark)

                    outputs = torch.clamp_min(outputs, -mean/scale)
                    loss = criterion(outputs, batch_y[:, :, -1:])

                train_loss.append(loss.item() * batch_x.size(0))
                num_data += batch_x.size(0)

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.sum(train_loss) / num_data
            vali_loss, vali_mae = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali MAE: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_mae))
            
            if self.args.wandb:
                wandb.log({
                    'Epoch': epoch + 1,
                    'Learning Rate': model_optim.param_groups[0]['lr'],
                    'Train Loss': train_loss,
                    'Valid Loss': vali_loss,
                    'Valid MAE' : vali_mae,
                })
            
            early_stopping(vali_mae, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            scheduler.step(epoch + 1)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        serials = []
        mean = pred_data.scaler.mean_[-1]
        scale = pred_data.scaler.scale_[-1]
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark, target_serial) in enumerate(tqdm(pred_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.long().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.long().to(self.device)

                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                
                outputs = outputs[:, 0, 0] # only last token
                outputs = outputs * scale + mean
                outputs = torch.clamp_min(outputs, 0)
                outputs = outputs.detach().cpu().numpy()
                
                preds.append(outputs)
                serials.append(target_serial)

        preds = np.array(preds).reshape(-1)
        serials = np.array(serials).reshape(-1)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        df_out = pd.DataFrame(np.stack((serials, np.round(preds, decimals=2)), axis=1), columns=['序號', '答案'])
        df_out['答案'] = df_out['答案'].astype(float)
        df_out.to_csv(folder_path + 'result_org.csv', index=False, encoding='utf-8-sig')

        df_out = df_out.groupby(['序號'], sort=False).mean()
        df_out['答案'] = df_out['答案'].round(2)
        df_out.to_csv(folder_path + 'result.csv', index=True, encoding='utf-8-sig')

        return
