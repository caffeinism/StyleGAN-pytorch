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

    print(config)
    
    trainer = Trainer(
        dataset_dir=config.dataset_dir, 
        generator_channels=config.generator_channels,
        discriminator_channels=config.discriminator_channels,
        nz=config.nz, 
        lr=config.lr, 
        betas=config.betas, 
        eps=config.eps,
        phase_iter=config.phase_iter,
        batch_size=config.batch_size,
    ) 
    trainer.run(config.log_iter)
 
if __name__ == '__main__':
    main()