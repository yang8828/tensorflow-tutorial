import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

img=tf.gfile.FastGFile('/home/yangqr8828/Desktop/Test_DL/mnist_test/image/timg.jpeg','r').read()

def distort_color(img,color_order=0):
	if color_order==0:
		img=tf.image.random_brightness(img,max_delta=32.0/255.0)
		img=tf.image.random_saturation(img,lower=0.5,upper=1.5)
		img=tf.image.random_hue(img,max_delta=0.2)
		img=tf.image.random_contrast(img,lower=0.5,upper=1.5)
	elif color_order==1:
		img=tf.image.random_saturation(img,lower=0.5,upper=1.5)		
		img=tf.image.random_brightness(img,max_delta=32.0/255.0)
		img=tf.image.random_contrast(img,lower=0.5,upper=1.5)
		img=tf.image.random_hue(img,max_delta=0.2)
	elif color_order==2:
		img=tf.image.random_hue(img,max_delta=0.2)		
		img=tf.image.random_contrast(img,lower=0.5,upper=1.5)
		img=tf.image.random_brightness(img,max_delta=32.0/255.0)
		img=tf.image.random_saturation(img,lower=0.5,upper=1.5)
	else:
		img=tf.image.random_contrast(img,lower=0.5,upper=1.5)
		img=tf.image.random_hue(img,max_delta=0.2)
		img=tf.image.random_saturation(img,lower=0.5,upper=1.5)
		img=tf.image.random_brightness(img,max_delta=32.0/255.0)
	return tf.clip_by_value(img,0.0,1.0)

def preprocess_for_train(image,height,width,bbox):
	if bbox==None:
		bbox=tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
	if image.dtype!=tf.float32:
		image=tf.image.convert_image_dtype(image,dtype=tf.float32)
	bbox_begin,bbox_size,_=tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
	distorted_image=tf.slice(image,	bbox_begin,bbox_size)
	distorted_image=tf.image.resize_images(distorted_image,[height,width],method=1)
	distorted_image=tf.image.random_flip_up_down(distorted_image)
	distorted_image=tf.image.random_flip_left_right(distorted_image)
	distorted_image=distort_color(distorted_image,np.random.randint(3))
	return distorted_image
	
with tf.Session() as sess:
	img_data=tf.image.decode_jpeg(img)
	boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
	for i in range(6):
		result=preprocess_for_train(img_data,299,299,boxes)
		plt.imshow(result.eval())
		plt.show()
