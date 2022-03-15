import torch
from torch import nn

from torch.nn import MaxPool2d
import numpy as np

import config
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from toolkit import ml, dataset

gpu = True


class LazyLoadDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        self.paths = []
        dct = dataset.get_all_classes_dict()
        for tag in dct:
            if int(tag) < config.classification_count:
                dir = f'data/{tag}_{dct[tag]}'
                for f in (ml.getAllFiles(dir)):
                    self.paths.append([f, int(tag)])
        print('data length', len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        a = self.paths[idx]
        # print(a)
        x, y = dataset.from_single_file(a[0], a[1])
        return x.reshape(x.shape[1:]), y.reshape(y.shape[1:])


if __name__ == '__main__':

    face_dataset = LazyLoadDataset()
    dataloader = DataLoader(face_dataset, batch_size=16,
                            shuffle=True, num_workers=0)


    class Classifier(nn.Module):
        def __init__(self, num_classes):
            super().__init__()

            self.conv = torch.nn.Sequential(
                nn.LazyConv2d(64, kernel_size=(3, 3)),
                nn.ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.LazyConv2d(64, kernel_size=(3, 3)),
                nn.ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.LazyConv2d(64, kernel_size=(3, 3)),
                nn.ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.LazyConv2d(64, kernel_size=(3, 3)),
                nn.ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.LazyConv2d(64, kernel_size=(3, 3)),
                nn.ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.LazyConv2d(64, kernel_size=(3, 3)),
                nn.ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=2),
            )

            self.fc = nn.Sequential(
                nn.LazyLinear(256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.1),
                nn.LazyLinear(num_classes),
            )

        def forward(self, x):
            # print('forwarding shape',x.shape)
            x = self.conv(x)
            x = torch.reshape(x, (x.shape[0], -1))
            x = self.fc(x)
            return x


    # data2, tags2 = dataset.from_types(['1_fish_fail'], 1)
    # data = torch.cat([data, data2])
    # tags = torch.cat([tags, tags2])


    model = Classifier(config.classification_count)

    print(model)

    # model.eval()

    if gpu:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    # model.eval()
    # 训练
    history = {'loss': []}

    for t in range(config.torch_train_epoch):
        # data, tags = dataset.from_just_class_nums([config.current_train_on])

        loss_avg = []
        for _, dda in enumerate(dataloader):
            data, tags = dda
            # print(x.shape)
            # print(y)

            x = data
            # print(x.shape)
            prediction = model(x)
            # print(prediction.shape)
            # loss = loss_func(prediction, torch.from_numpy(target).type(torch.long).cuda())
            loss = loss_func(prediction, tags)
            loss_avg.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('iter', t, np.average(loss_avg))
        history['loss'].append(np.average(loss_avg))

    model_scripted = torch.jit.script(model)  # Export to TorchScript
    # model_scripted.save('model/xiaobo_status_check.pt')  # Save
    model_scripted.save(config.train_save_path)  # Save

    model.eval()
    # print(model())
    print(torch.argmax(model(dataset.from_screen())))

    pd.DataFrame(history).plot(figsize=(8, 5))
    plt.legend()
    plt.show()
