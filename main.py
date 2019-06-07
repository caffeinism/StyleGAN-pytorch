from trainer import Trainer
import argparse
from config import Config
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

parser = argparse.ArgumentParser("train")
parser.add_argument('--config', type=str, required=True)
args, _ = parser.parse_known_args()
        
def main(): 
    # pylint: disable=no-member
    config = Config(args.config)
    dataset = dset.ImageFolder(root=config.dataset_dir, transform=transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    trainer = Trainer(
        generator_channels=config.generator_channels,
        discriminator_channels=config.discriminator_channels,
        nz=config.nz, 
        lr=config.lr, 
        betas=config.betas, 
        eps=config.eps
    ) 
    trainer.run(dataloader)
 
if __name__ == '__main__':
    main()