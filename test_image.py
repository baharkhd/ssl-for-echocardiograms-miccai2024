import numpy as np
import PIL.Image as Image
from scipy.io import loadmat
import matplotlib.pyplot as plt

def LoadImageFeature(file_path):
    im = Image.open(file_path)
    im = np.asarray(im)
    im = im[:,:, np.newaxis] #make it (64, 64, 1) 1:channel
    return im

im = LoadImageFeature('/home/baharkhd/ssl-for-echocardiograms-miccai2024/data/TMED/approved_users_only/view_labeled_set/labeled/5s1_0.png')

print(type(im), im.shape)

our_im_path = '/mnt/nas-server/published/vaseli_ProtoASNet_MICCAI2023/data_removelater/as_tom/plax/preprocessed/001_1.2.840.113619.2.185.2838.1276067673.0.679.512.mat'
mat_data = loadmat(our_im_path)

print(type(mat_data['cine']), mat_data['cine'].shape)

plt.imshow(im, cmap='gray')
plt.savefig('test_img2.png')

