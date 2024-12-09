from data_provider.data_loader import Dataset_Pred, Dataset_Tbrain
from torch.utils.data import DataLoader

data_dict = {
    'Tbrain': Dataset_Tbrain,
}


def data_provider(args, flag, data_set=None):
    Data = data_dict[args.data]
    Data_scaler = Data
    timeenc = 0 if args.embed != 'timeF' else 1
    
    if flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Dataset_Pred
    else:
        raise NotImplementedError
    
    if data_set is None:
        scaler = Data_scaler(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='train',
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=args.freq,
        ).scaler

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=args.freq,
            scaler=scaler,
        )

    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
    return data_set, data_loader
