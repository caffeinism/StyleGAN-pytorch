from trainer import Trainer

def main():
    dataloader = NotImplemented
    trainer = Trainer(nz=512, lr=0.001, betas=(0.5, 0.999))
    trainer.run(dataloader)
 
if __name__ == '__main__':
    main()