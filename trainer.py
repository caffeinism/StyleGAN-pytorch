from networks import Generator, Discriminator

class Trainer:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
    
    def run(self):
        pass