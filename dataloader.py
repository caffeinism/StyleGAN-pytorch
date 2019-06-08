import torch
import math
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class Dataloader:
    def __init__(self, dataset_dir, batch_sizes, max_tick):
        self.dataset_dir = dataset_dir
        self.batch_sizes = batch_sizes
        self.img_size = 2
        self.grow()
        self.max_tick = max_tick

    def __iter__(self):
        tick = 0
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        while True:
            for data in dataloader:
                if tick >= self.max_tick:
                    return
                
                yield data, tick
                tick += self.batch_size

    def grow(self):
        self.img_size *= 2
        self.batch_size = self.batch_sizes[str(self.img_size)]

        self.dataset = ImageFolder(root=self.dataset_dir, transform=transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ]))

    def len(self):
        return math.ceil(self.max_tick / self.batch_size)