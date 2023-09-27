import numpy as np

def conv(_img,kernel_):
    iw , ih , ch_ = _img.shape
    kw , kh = kernel_.shape
    fw , fh = iw - kw + 1 , ih - kh + 1
    op_ = np.zeros((fw,fh))
    
    for i in range(fw):
        for j in range(fh):
            for k in range(ch_):
                patch = _img[i:i+kw,j:j+kh,k]
                op_[i,j] += np.dot(patch.flatten(),kernel_.flatten())
    return op_
img_ = np.random.random((224,224,3))
kernel_ = np.random.random((3,3))
x = conv(img_,kernel_)
