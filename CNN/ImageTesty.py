import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
with open('/timg.jpg', 'rb') as f:
	img=f.read()
with tf.Session() as sess:
	img_data=tf.image.decode_jpeg(img)
	print(img_data.eval())
