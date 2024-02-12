import numpy as np
import PIL.Image as Image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf

#def LoadImageFeature(file_path):
#    im = Image.open(file_path)
#    im = np.asarray(im)
#    im = im[:,:, np.newaxis] #make it (64, 64, 1) 1:channel
#    return im

#im = LoadImageFeature('/home/baharkhd/ssl-for-echocardiograms-miccai2024/data/TMED/approved_users_only/view_labeled_set/labeled/5s1_0.png')

#print(type(im), im.shape)

#our_im_path = '/mnt/nas-server/published/vaseli_ProtoASNet_MICCAI2023/data_removelater/as_tom/plax/preprocessed/001_1.2.840.113619.2.185.2838.1276067673.0.679.512.mat'
#mat_data = loadmat(our_im_path)

#print(type(mat_data['cine']), mat_data['cine'].shape)

#plt.imshow(im, cmap='gray')
#plt.savefig('test_img2.png')


tfrecord_file = '/home/baharkhd/ssl-for-echocardiograms-miccai2024/results/TMED-18-18/fold0/train_DIAGNOSIS.tfrecord'
dataset = tf.data.TFRecordDataset(tfrecord_file)

# Define a function to parse the records
def parse_tfrecord_fn(example_proto):
    feature_description = {
        'feature1': tf.io.FixedLenFeature([], tf.int64),
        'feature2': tf.io.FixedLenFeature([], tf.float32),
        # Add more features based on your data
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def parse_tfrecord_fn(example_proto):
    feature_description = {
        'feature1': tf.io.FixedLenFeature([], tf.int64),
        'feature2': tf.io.FixedLenFeature([], tf.float32),
        # Add more features based on your data
    }
    return tf.io.parse_single_example(example_proto, feature_description)

# Map the parsing function to the dataset
parsed_dataset = dataset.map(parse_tfrecord_fn)

# Define additional preprocessing steps or transformations
# For example, you can batch the dataset, shuffle it, etc.
batched_dataset = parsed_dataset.batch(32)
shuffled_dataset = batched_dataset.shuffle(buffer_size=1000)

# Iterate over the dataset
for record in shuffled_dataset.take(5):
    print(record)

