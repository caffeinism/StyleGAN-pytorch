from networks import Generator, Discriminator
import torch
import torch.optim as optim
import torch.nn.functional as F

alpha = 1.0 # for test
class Trainer:
    def __init__(self, generator_channels, discriminator_channels, nz, lr, betas, eps):
        self.nz = nz

        self.generator = Generator(generator_channels, nz).cuda()
        self.discriminator = Discriminator(discriminator_channels).cuda()

        self.optimizer_g = optim.Adam(params=self.generator.parameters(), lr=lr, betas=betas)
        self.optimizer_d = optim.Adam(params=self.discriminator.parameters(), lr=lr, betas=betas)
    
    def generator_trainloop(self, batch_size):
        z = torch.randn(batch_size, self.nz).cuda()

        fake = self.generator(z, alpha=alpha)
        d_fake = self.discriminator(fake, alpha=alpha)
        loss = F.softplus(-d_fake).mean()

        self.optimizer_g.zero_grad()
        loss.backward()
        self.optimizer_g.step()
    
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
        

    def run(self, dataloader):
        global_iter = 0
        while True:
            for data, _ in dataloader:
                real = data.cuda()

                self.discriminator_trainloop(real)
                self.generator_trainloop(real.size(0))

                global_iter += 1
