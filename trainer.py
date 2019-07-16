from networks import Generator, Discriminator
import torch
import torch.optim as optim
import torch.nn.functional as F
import tf_recorder as tensorboard
from tqdm import tqdm
from dataloader import Dataloader
from torch.autograd import grad
import amp_support as amp
import random

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class Trainer:
    def __init__(self, dataset_dir, generator_channels, discriminator_channels, nz, style_depth, lrs, betas, eps, 
                 phase_iter, batch_size, n_cpu, opt_level):
        self.nz = nz
        self.dataloader = Dataloader(dataset_dir, batch_size, phase_iter * 2, n_cpu)

        self.generator = Generator(generator_channels, nz, style_depth).cuda()
        self.discriminator = Discriminator(discriminator_channels).cuda()

        self.tb = tensorboard.tf_recorder('StyleGAN')

        self.phase_iter = phase_iter
        self.lrs = lrs
        self.betas = betas

        self.opt_level = opt_level

    def generator_trainloop(self, batch_size, alpha):
        requires_grad(self.generator, True)
        requires_grad(self.discriminator, False)
        
        # mixing regularization
        if random.random() < 0.9:
            z = [torch.randn(batch_size, self.nz).cuda(),
                 torch.randn(batch_size, self.nz).cuda()]
        else:
            z = torch.randn(batch_size, self.nz).cuda()

        fake = self.generator(z, alpha=alpha)
        d_fake = self.discriminator(fake, alpha=alpha)
        loss = F.softplus(-d_fake).mean()

        self.optimizer_g.zero_grad()
        with amp.scale_loss(loss, self.optimizer_g) as scaled_loss:
            scaled_loss.backward()
        self.optimizer_g.step()

        return loss.item()
    
    def discriminator_trainloop(self, real, alpha):
        requires_grad(self.generator, False)
        requires_grad(self.discriminator, True)

        real.requires_grad = True
        self.optimizer_d.zero_grad()

        d_real = self.discriminator(real, alpha=alpha)
        loss_real = F.softplus(-d_real).mean()
        with amp.scale_loss(loss_real, self.optimizer_d) as scaled_loss_real:
            scaled_loss_real.backward(retain_graph=True)

        grad_real = grad(
            outputs=d_real.sum(), inputs=real, create_graph=True
        )[0]
        grad_penalty = (
            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        ).mean()
        grad_penalty = 10 / 2 * grad_penalty
        with amp.scale_loss(grad_penalty, self.optimizer_d) as scaled_grad_penalty:
            scaled_grad_penalty.backward()
        
        if random.random() < 0.9:
            z = [torch.randn(real.size(0), self.nz).cuda(),
                 torch.randn(real.size(0), self.nz).cuda()]
        else:
            z = torch.randn(real.size(0), self.nz).cuda()

        fake = self.generator(z, alpha=alpha)
        d_fake = self.discriminator(fake, alpha=alpha)
        loss_fake = F.softplus(d_fake).mean()
        with amp.scale_loss(loss_fake, self.optimizer_d) as scaled_loss_fake:
            scaled_loss_fake.backward()

        loss = scaled_loss_real + scaled_loss_fake + scaled_grad_penalty

        self.optimizer_d.step()
        
        return loss.item(), (d_real.mean().item(), d_fake.mean().item())

    def run(self, log_iter, checkpoint):
        global_iter = 0
        
        test_z = torch.randn(4, self.nz).cuda()

        if checkpoint:
            self.load_checkpoint(checkpoint)
        else:
            self.grow()

        while True:
            print('train {}X{} images...'.format(self.dataloader.img_size, self.dataloader.img_size))
            for iter, ((data, _), n_trained_samples) in enumerate(tqdm(self.dataloader), 1):
                real = data.cuda()
                alpha = min(1, n_trained_samples / self.phase_iter) if self.dataloader.img_size > 8 else 1

                loss_d, (real_score, fake_score) = self.discriminator_trainloop(real, alpha)
                loss_g = self.generator_trainloop(real.size(0), alpha)

                if global_iter % log_iter == 0:
                    self.log(loss_d, loss_g, real_score, fake_score, test_z, alpha)

                # save 3 times during training
                if iter % (len(self.dataloader) // 4 + 1) == 0:
                    self.save_checkpoint(n_trained_samples)

                global_iter += 1
                self.tb.iter(data.size(0))

            self.save_checkpoint()
            self.grow()


    def log(self, loss_d, loss_g, real_score, fake_score, test_z, alpha):
        with torch.no_grad():
            fake = self.generator(test_z, alpha=alpha)
            fake = (fake + 1) * 0.5
            fake = torch.clamp(fake, min=0.0, max=1.0)

        self.tb.add_scalar('loss_d', loss_d)
        self.tb.add_scalar('loss_g', loss_g)
        self.tb.add_scalar('real_score', real_score)
        self.tb.add_scalar('fake_score', fake_score)
        self.tb.add_images('fake', fake)

    def grow(self):
        self.discriminator.grow()
        self.generator.grow()
        self.dataloader.grow()
        self.generator.cuda()
        self.discriminator.cuda()
        self.tb.renew('{}x{}'.format(self.dataloader.img_size, self.dataloader.img_size))

        self.lr = self.lrs.get(str(self.dataloader.img_size), 0.001)
        self.style_lr = self.lr * 0.01

        self.optimizer_d = optim.Adam(params=self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_g = optim.Adam([
                {'params': self.generator.model.parameters(), 'lr':self.lr},
                {'params': self.generator.style_mapper.parameters(), 'lr': self.style_lr},
            ],
            betas=self.betas
        )

        [self.generator, self.discriminator], [self.optimizer_g, self.optimizer_d] = amp.initialize(
            [self.generator, self.discriminator], 
            [self.optimizer_g, self.optimizer_d],
            opt_level=self.opt_level
        )    

    def save_checkpoint(self, tick='last'):
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'generator_optimizer': self.optimizer_g.state_dict(),
            'discriminator_optimizer': self.optimizer_d.state_dict(),
            'img_size': self.dataloader.img_size,
            'tick': tick,
        }, 'checkpoints/{}x{}_{}.pth'.format(self.dataloader.img_size, self.dataloader.img_size, tick))

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)

        print('load {}x{} checkpoint'.format(checkpoint['img_size'], checkpoint['img_size']))
        while self.dataloader.img_size < checkpoint['img_size']:
            self.grow()

        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_g.load_state_dict(checkpoint['generator_optimizer'])
        self.optimizer_d.load_state_dict(checkpoint['discriminator_optimizer'])

        if checkpoint['tick'] == 'last':
            self.grow()
        else:
            self.dataloader.set_checkpoint(checkpoint['tick'])
            self.tb.iter(checkpoint['tick'])