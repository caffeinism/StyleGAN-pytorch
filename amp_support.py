import contextlib
try:
    from apex import amp
except ImportError as e:
    amp = None

def scale_loss(*args, **kwargs):
    if amp:
        return amp.scale_loss(*args, **kwargs)
    else:
        return dummy_scale_loss(*args, **kwargs)

def initialize(models, optimizers, *args, **kwargs):
    if amp:
        return amp.initialize(models, optimizers, *args, **kwargs)
    else:
        return models, optimizers

@contextlib.contextmanager
def dummy_scale_loss(loss, *args, **kwargs):
    yield loss
