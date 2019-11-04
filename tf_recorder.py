from tensorboardX import SummaryWriter
import os, sys
import os.path

# https://github.com/nashory/pggan-pytorch/blob/master/tf_recorder.py
class tf_recorder:
    def __init__(self, network_name, log_dir):
        os.system('mkdir -p {}'.format(log_dir))
        for i in range(1000):
            self.targ = os.path.join(log_dir, '{}_{}'.format(network_name, i))
            if not os.path.exists(self.targ):
                self.writer = SummaryWriter(self.targ)
                break
    
    def renew(self, subname):
        self.writer = SummaryWriter('{}_{}'.format(self.targ, subname))
        self.niter = 0
                
    def add_scalar(self, index, val):
        self.writer.add_scalar(index, val, self.niter)

    def add_scalars(self, index, group_dict):
        self.writer.add_scalar(index, group_dict, self.niter)

    def add_images(self, tag, images):
        self.writer.add_images(tag, images, self.niter)

    def iter(self, tick=1):
        self.niter += tick

