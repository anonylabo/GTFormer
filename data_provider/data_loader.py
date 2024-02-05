import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, flag, args, od_matrix):
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.args = args

        self.seq_len = args.seq_len
        day_steps = {"60min": 24, "45min": 32, "30min": 48, "15min": 96}
        self.day_step = day_steps[args.sample_time]
        self.__read_data__(od_matrix)

    def __read_data__(self, od_matrix):
        # Change the extraction period according to train, valid, test
        data_days = len(od_matrix) // self.day_step
        border1s = [0, self.day_step * int((data_days - 10) * 0.8), self.day_step * int(data_days - 10)]
        border2s = [
            self.day_step * int((data_days - 10) * 0.8),
            self.day_step * int(data_days - 10),
            self.day_step * data_days,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data = torch.tensor(od_matrix[border1:border2]).to(self.args.dtype)

    def __getitem__(self, index):
        seq_x = self.data[index : index + self.seq_len]
        seq_y = self.data[index + self.seq_len : index + self.seq_len + 1]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data) - self.seq_len - 1


def data_provider(flag, args, od_matrix):
    shuffle_flag = False
    drop_last = False
    batch_size = args.batch_size

    data_set = MyDataset(flag, args, od_matrix)

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, drop_last=drop_last, num_workers=2)

    return data_loader
