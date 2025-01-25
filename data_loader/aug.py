import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq

class Reshape(object):
    def __call__(self, seq):
        return seq.transpose()


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)



class Normalize(object):
    def __init__(self, norm_type="0-1"):
        assert norm_type in ["0-1", "-1-1", "mean-std", "None"], \
            f"Normalization should be '0-1', '-1-1', 'mean-std', or 'None', but got {norm_type}"
        self.type = norm_type
        
    def __call__(self, seq):
        if self.type == "0-1":
            seq = (seq - seq.min()) / (seq.max() - seq.min())
        elif self.type == "-1-1":
            seq = 2 * (seq - seq.min()) / (seq.max() - seq.min()) - 1
        elif self.type == "mean-std":
            seq = (seq - seq.mean()) / seq.std()
        elif self.type == "None":
            pass
        
        
        return seq

