from networks import Generator, Discriminator
import torch
import torch.optim as optim
import torch.nn.functional as F

class Trainer:
    def __init__(self, nz, lr, betas):
        self.nz = nz

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.optimizer_g = optim.Adam(params=self.generator.parameters(), lr=lr, betas=betas)
        self.optimizer_d = optim.Adam(params=self.discriminator.parameters(), lr=lr, betas=betas)
    
    def generator_trainloop(self, batch_size):
        z = torch.randn(batch_size, self.nz)

        fake = self.generator(z)
        d_fake = self.discriminator(fake)
        loss = F.softplus(-d_fake).mean()

        self.optimizer_g.zero_grad()
        loss.backward()
        self.optimizer_g.step()
    
    def discriminator_trainloop(self, real):
        d_real = self.discriminator(real)
        loss_real = F.softplus(-d_real).mean()

        z = torch.randn(real.size(0), self.nz)

        fake = self.generator(z)
        d_fake = self.discriminator(fake)
        loss_fake = F.softplus(d_fake).mean()

        loss = loss_real + loss_fake

        self.optimizer_d.zero_grad()
        loss.backward()
        self.optimizer_d.step()
        

    def run(self, dataloader):
        global_iter = 0
        while True:
            for data in dataloader:
                real = data.cuda()

                self.discriminator_trainloop(real)
                self.generator_trainloop(real.size(0))

                global_iter += 1
