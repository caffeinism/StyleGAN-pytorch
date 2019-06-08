from networks import Generator, Discriminator
import torch
import torch.optim as optim
import torch.nn.functional as F
import tf_recorder as tensorboard
from tqdm import tqdm

alpha = 1.0 # for test
class Trainer:
    def __init__(self, generator_channels, discriminator_channels, nz, lr, betas, eps):
        self.nz = nz

        self.generator = Generator(generator_channels, nz).cuda()
        self.discriminator = Discriminator(discriminator_channels).cuda()

        self.optimizer_g = optim.Adam(params=self.generator.parameters(), lr=lr, betas=betas)
        self.optimizer_d = optim.Adam(params=self.discriminator.parameters(), lr=lr, betas=betas)
    
        self.tb = tensorboard.tf_recorder('StyleGAN')

    def generator_trainloop(self, batch_size):
        z = torch.randn(batch_size, self.nz).cuda()

        fake = self.generator(z, alpha=alpha)
        d_fake = self.discriminator(fake, alpha=alpha)
        loss = F.softplus(-d_fake).mean()

        self.optimizer_g.zero_grad()
        loss.backward()
        self.optimizer_g.step()

        return loss.item()
    
    def discriminator_trainloop(self, real):
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

    def run(self, dataloader, log_iter):
        global_iter = 0
        while True:
            for data, _ in tqdm(dataloader):
                real = data.cuda()

                loss_d = self.discriminator_trainloop(real)
                loss_g = self.generator_trainloop(real.size(0))

                if global_iter % log_iter == 0:
                    self.log(loss_d, loss_g)

                global_iter += 1

    def log(self, loss_d, loss_g):
        with torch.no_grad():
            z = torch.randn(4, self.nz).cuda() # TODO: 4 -> batch_size
            fake = self.generator(z, alpha=alpha)

        self.tb.add_scalar('loss_d', loss_d)
        self.tb.add_scalar('loss_g', loss_g)
        self.tb.add_images('fake', fake)
        self.tb.iter()

        