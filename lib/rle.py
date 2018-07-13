import numpy as np
def rle_encode(x):
    x_flatten = x.ravel()
    t = np.concatenate([[2], x_flatten, [2]])
    idx = np.where(t[1:]!=t[:-1])[0]
    rl = idx[1:]-idx[:-1]
    if x_flatten[0]==1:
        rl = np.append([0], rl)
    return rl, x.shape

def rle_decode(x, shape):
    value = 0
    decoded = []
    for l in x:
        decoded += [value]*l
        value = (value+1)&1
    return np.asarray(decoded).reshape(shape)
if __name__ == '__main__':
    for _ in range(1000):
        x = np.random.binomial(1, 0.5, 100).reshape(10,10)
        x_rle, shape = rle_encode(x)
        x_h = rle_decode(x_rle, shape)
        print(x)
        print(x_h)
        assert (x_h==x).all()
    print('Self test passed.')
