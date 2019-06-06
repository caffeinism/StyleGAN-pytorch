import yaml
from argparse import Namespace

class Config(Namespace): 
    def __init__(self, filename):
        config = yaml.load(open(filename, 'r'))
        super(Config, self).__init__(**config)