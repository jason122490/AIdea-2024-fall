import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import glob
from tqdm import tqdm

warnings.filterwarnings('ignore')

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path=None,
                 target='Power(mW)', scale=True, timeenc=0, freq='h', scaler=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = 'Power(mW)'
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.scaler = scaler
        self.__read_data__()

    def __read_data__(self):
        # target data
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path)).astype(str)
        '''
        df_raw.columns: ['序號', '答案']
        '''
        df_stamp = pd.DataFrame()
        df_stamp['DateTime'] = pd.to_datetime(df_raw['序號'].str.slice(0, 12), format='%Y%m%d%H%M').dt.floor('min')
        df_stamp['LocationCode'] = df_raw['序號'].str.slice(12, 14).astype(int)
        df_stamp['serial'] = df_raw['序號']

        offset_df_stamp = df_stamp.copy()
        for i in range(9):
            offset_df_stamp['DateTime'] = offset_df_stamp['DateTime'] + pd.DateOffset(minutes=1)
            df_stamp = pd.concat([df_stamp, offset_df_stamp])

        df_stamp['month'] = df_stamp.DateTime.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.DateTime.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.DateTime.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.DateTime.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.DateTime.apply(lambda row: row.minute, 1)

        df_stamp = df_stamp[['DateTime', 'month', 'day', 'weekday', 'hour', 'minute', 'LocationCode', 'serial']]
        df_stamp = df_stamp.reset_index(drop=True)

        self.target_time = df_stamp['DateTime']
        self.target_stamp = df_stamp.drop(columns=['DateTime', 'serial']).values
        self.target_serial = df_stamp['serial'].values

        # knowledge data
        df_raw = pd.concat([pd.read_csv(file) for file in glob.glob(os.path.join(self.root_path, '*.csv'))]).reset_index(drop=True)
        
        '''
        df_raw.columns: ['DateTime', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('DateTime')
        df_raw = df_raw[['DateTime'] + cols + [self.target]]

        df_stamp = df_raw[['DateTime']]
        df_stamp['DateTime'] = pd.to_datetime(df_stamp.DateTime).dt.floor('min')
        df_stamp['month'] = df_stamp.DateTime.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.DateTime.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.DateTime.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.DateTime.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.DateTime.apply(lambda row: row.minute, 1)
        df_stamp['LocationCode'] = df_raw['LocationCode']
        df_stamp = df_stamp.sort_values(by=['DateTime', 'LocationCode'])

        df_data = df_raw.iloc[df_stamp.index]

        self.data = df_data.drop(columns=['DateTime', 'LocationCode']).values
        self.data_stamp = df_stamp.drop(columns='DateTime').values
        self.data_time = pd.DatetimeIndex(df_stamp['DateTime'])

        if self.scale:
            self.data = self.scaler.transform(self.data)
        else:
            self.data = df_data.values

    def __getitem__(self, index):
        target_time = self.target_time[index]
        target_serial = self.target_serial[index]
        target_index = self.data_time.get_indexer_non_unique([target_time])[0][0]

        seq_y = np.zeros((self.pred_len, 6))
        seq_y_mark = self.target_stamp[index:index + self.pred_len]

        if target_index - self.seq_len < 0:
            seq_x = self.data[0 : self.seq_len]
            seq_x_mark = self.data_stamp[0 : self.seq_len]
        elif target_index + self.seq_len // 2 > self.data.shape[0]:
            seq_x = self.data[-self.seq_len:]
            seq_x_mark = self.data_stamp[-self.seq_len:]
        else:
            seq_x = self.data[target_index - self.seq_len // 2 : target_index + self.seq_len // 2]
            seq_x_mark = self.data_stamp[target_index - self.seq_len // 2 : target_index + self.seq_len // 2]

        return seq_x, seq_x_mark, seq_y, seq_y_mark, target_serial

    def __len__(self):
        return len(self.target_time)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Tbrain(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path=None,
                 target='Power(mW)', scale=True, timeenc=0, freq='h', scaler=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'val']
        self.flag = flag

        self.features = features
        self.target = 'Power(mW)'
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = glob.glob(os.path.join(self.root_path, '*.csv'))
        self.scaler = scaler
        if self.scaler is None and self.scale:
            self.__scaler__()
        else:
            self.__read_data__()

    def __scaler__(self):
        self.scaler = StandardScaler()
        df_raw = pd.concat([pd.read_csv(file) for file in self.data_path])

        '''
        df_raw.columns: ['DateTime', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('DateTime')
        cols.remove('LocationCode')
        df_data = df_raw[cols + [self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            print('scaler name:  ', df_data.columns.values)
            print('scaler mean:  ', self.scaler.mean_)
            print('scaler scale: ', self.scaler.scale_)

    def __read_data__(self):
        df_raw = pd.concat([pd.read_csv(file) for file in self.data_path]).reset_index(drop=True)
        
        '''
        df_raw.columns: ['DateTime', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('DateTime')
        df_raw = df_raw[['DateTime'] + cols + [self.target]]

        df_stamp = df_raw[['DateTime']]
        df_stamp['DateTime'] = pd.to_datetime(df_stamp.DateTime).dt.floor('min')
        df_stamp['month'] = df_stamp.DateTime.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.DateTime.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.DateTime.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.DateTime.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.DateTime.apply(lambda row: row.minute, 1)
        df_stamp['LocationCode'] = df_raw['LocationCode']
        df_stamp = df_stamp.sort_values(by=['DateTime', 'LocationCode'])

        df_data = df_raw.iloc[df_stamp.index]
        df_stamp['group_idx'] = df_stamp.groupby(['month', 'day', 'LocationCode']).ngroup()

        num_date = df_stamp['group_idx'].nunique()
        num_train = int(num_date * 0.9)

        shuffle_index = np.random.RandomState(seed=1224).permutation(num_date)
        train_index = shuffle_index[:num_train]
        valid_index = shuffle_index[num_train:]

        if self.flag == 'train':
            mask = df_stamp['group_idx'].isin(train_index)

            self.data = df_data[mask].drop(columns=['DateTime', 'LocationCode']).values
            self.data_stamp = df_stamp[mask].drop(columns='DateTime').values
            self.target_index = np.arange(len(self.data))
        elif self.flag == 'val':
            mask = df_stamp['group_idx'].isin(valid_index)

            self.data = df_data.drop(columns=['DateTime', 'LocationCode']).values
            self.data_stamp = df_stamp.drop(columns='DateTime').values
            self.target_index = df_stamp[mask].index
        else:
            raise ValueError('flag should be train or val')
        
        # print(self.data_stamp)
        # print(self.data)

        if self.scale:
            self.data = self.scaler.transform(self.data)
        else:
            self.data = df_data.values

        start_position = []
        end_position = []
        for target_index in tqdm(self.target_index):
            mask = self.data_stamp[:, -1] != self.data_stamp[target_index, -1]
            cumsum = mask.cumsum()
            position = cumsum[target_index]

            if position - self.seq_len // 2 < 0:
                start_position.append(0)
                end_position.append(np.searchsorted(cumsum, self.seq_len) + 1)
            elif position + self.seq_len // 2 > cumsum[-1]:
                start_position.append(np.searchsorted(cumsum, cumsum[-1] - self.seq_len + 1))
                end_position.append(self.data.shape[0])
            else:
                start_position.append(np.searchsorted(cumsum, position - self.seq_len // 2 + 1))
                end_position.append(np.searchsorted(cumsum, position + self.seq_len // 2) + 1)

        self.start_position = np.array(start_position)
        self.end_position = np.array(end_position)
        
    def __getitem__(self, index):
        target_index = self.target_index[index]
        start_position = self.start_position[index]
        end_position = self.end_position[index]
        
        mask = self.data_stamp[start_position : end_position, -1] != self.data_stamp[target_index, -1]
        data = self.data[start_position : end_position]
        stamp = self.data_stamp[start_position : end_position]

        seq_x = data[mask]
        seq_x_mark = stamp[mask, :-1]
        
        seq_y = self.data[target_index : target_index + self.pred_len]
        seq_y_mark = self.data_stamp[target_index : target_index + self.pred_len, :-1]

        return seq_x, seq_x_mark, seq_y, seq_y_mark

    def __len__(self):
        return len(self.target_index)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
