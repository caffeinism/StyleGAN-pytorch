import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class Dataloader:
    def __init__(self, dataset_dir, batch_sizes, max_tick, n_cpu):
        self.dataset_dir = dataset_dir
        self.batch_sizes = batch_sizes
        self.img_size = 4
        self.max_tick = max_tick
        self.checkpoint = 0
        self.n_cpus = n_cpu

    def __iter__(self):
        return DataIter(self.dataset, self.batch_size, self.max_tick, self.checkpoint, self.n_cpu)
    
    def set_checkpoint(self, checkpoint_tick):
        self.checkpoint = checkpoint_tick

    def grow(self):
        self.checkpoint = 0
        self.img_size *= 2
        self.batch_size = self.batch_sizes[str(self.img_size)]
        self.n_cpu = self.n_cpus[str(self.img_size)]

        self.dataset = ImageFolder(root=self.dataset_dir, transform=transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    def __len__(self):
        return (self.max_tick - self.checkpoint) // self.batch_size

class DataIter:
    def __init__(self, dataset, batch_size, max_tick, checkpoint, n_cpu):
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True, drop_last=True, num_workers=n_cpu,
        )
        self.iter = iter(self.dataloader)
        self.tick = self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.max_tick = max_tick

    def __next__(self):
        if self.tick >= self.max_tick:
            raise StopIteration

        try:
            data = next(self.iter)
        except StopIteration as e:
            self.iter = iter(self.dataloader)
            data = next(self.iter)
        
        self.tick += self.batch_size
        
        return data, self.tick

    def __len__(self):
        return (self.max_tick - self.checkpoint) // self.batch_size

