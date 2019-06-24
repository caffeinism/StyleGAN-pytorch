import argparse
from config import Config
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

parser = argparse.ArgumentParser("train")
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--type', type=str, default='train')
parser.add_argument('--checkpoint', type=str, default='')
args, _ = parser.parse_known_args()
        
def main(): 
    # pylint: disable=no-member
    config = Config(args.config)

    print(config)

    if args.type == 'train':
        from trainer import Trainer
        trainer = Trainer(
            dataset_dir=config.dataset_dir, 
            generator_channels=config.generator_channels,
            discriminator_channels=config.discriminator_channels,
            nz=config.nz, 
            style_depth=config.style_depth,
            lrs=config.lrs, 
            betas=config.betas, 
            eps=config.eps,
            phase_iter=config.phase_iter,
            batch_size=config.batch_size,
            n_cpu=config.n_cpu,
        ) 
        trainer.run(log_iter=config.log_iter, checkpoint=args.checkpoint)
    elif args.type == 'inference':
        from inferencer import Inferencer
        inferencer = Inferencer(
            generator_channels=config.generator_channels,
            nz=config.nz, 
            style_depth=config.style_depth,
        )
        inferencer.inference(n=8)
    else:
        raise NotImplementedError
    
if __name__ == '__main__':
    main()