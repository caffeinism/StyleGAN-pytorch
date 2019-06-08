from networks import Generator, Discriminator
import torch
import torch.optim as optim
import torch.nn.functional as F
import tf_recorder as tensorboard
from tqdm import tqdm
from dataloader import Dataloader

class Trainer:
    def __init__(self, dataset_dir, generator_channels, discriminator_channels, nz, lr, betas, eps, 
                 phase_iter, batch_size):
        self.nz = nz
        self.dataloader = Dataloader(dataset_dir, batch_size, phase_iter * 2)

        self.generator = Generator(generator_channels, nz).cuda()
        self.discriminator = Discriminator(discriminator_channels).cuda()
        
        self.optimizer_g = optim.Adam(params=self.generator.parameters(), lr=lr, betas=betas)
        self.optimizer_d = optim.Adam(params=self.discriminator.parameters(), lr=lr, betas=betas)
    
        self.tb = tensorboard.tf_recorder('StyleGAN')

        self.phase_iter = phase_iter

    def generator_trainloop(self, batch_size, alpha):
        z = torch.randn(batch_size, self.nz).cuda()

        fake = self.generator(z, alpha=alpha)
        d_fake = self.discriminator(fake, alpha=alpha)
        loss = F.softplus(-d_fake).mean()

        self.optimizer_g.zero_grad()
        loss.backward()
        self.optimizer_g.step()

        return loss.item()
    
    def discriminator_trainloop(self, real, alpha):
        d_real = self.discriminator(real, alpha=alpha)
        loss_real = F.softplus(-d_real).mean()

        z = torch.randn(real.size(0), self.nz).cuda()

        fake = self.generator(z, alpha=alpha)
        d_fake = self.discriminator(fake, alpha=alpha)
        loss_fake = F.softplus(d_fake).mean()

        loss = loss_real + loss_fake

        self.optimizer_d.zero_grad()
        loss.backward()
        self.optimizer_d.step()
        
        return loss.item()

    def run(self, log_iter):
        global_iter = 0
        
        while True:
            self.discriminator.grow()
            self.generator.grow()
            self.dataloader.grow()
            self.generator.cuda()
            self.discriminator.cuda()

            print('train {}X{} images...'.format(self.dataloader.img_size, self.dataloader.img_size))
            for (data, _), n_trained_samples in tqdm(self.dataloader):
                real = data.cuda()
                alpha = min(1, n_trained_samples / self.phase_iter)

                loss_d = self.discriminator_trainloop(real, alpha)
                loss_g = self.generator_trainloop(real.size(0), alpha)

                if global_iter % log_iter == 0:
                    self.log(loss_d, loss_g)

                global_iter += 1


    def log(self, loss_d, loss_g):
        with torch.no_grad():
            z = torch.randn(4, self.nz).cuda() # TODO: 4 -> batch_size
            fake = self.generator(z, alpha=1)

        self.tb.add_scalar('loss_d', loss_d)
        self.tb.add_scalar('loss_g', loss_g)
        self.tb.add_images('fake', fake)
        self.tb.iter()

        