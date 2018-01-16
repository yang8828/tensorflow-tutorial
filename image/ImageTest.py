import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.platform import gfile
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
img=tf.gfile.FastGFile('/home/yangqr8828/Desktop/Test_DL/mnist_test/image/timg.jpeg','r').read()
with tf.Session() as sess:
	img_data=tf.image.decode_jpeg(img)
	#print img_data.eval()
	
	#img_data=tf.image.convert_image_dtype(img_data,dtype=tf.float32)
	#encoded_img=tf.image.encode_jpeg(img_data)
	#print img_data.eval()
	#tf.gfile.FastGFile('/home/yangqr8828/Desktop/Test_DL/mnist_test/image/timg_dt.jpeg','wb').write(encoded_img.eval)

	#resize=tf.image.resize_images(img_data,[200,200],method=1)
	cropped=tf.image.resize_image_with_crop_or_pad(img_data,300,300)
	padded =tf.image.resize_image_with_crop_or_pad(img_data,1000,1000)
	central_crop=tf.image.central_crop(img_data,0.7)
	flip_up_down=tf.image.flip_up_down(img_data)
	flip_left_right=tf.image.flip_left_right(img_data)
	transposed=tf.image.transpose_image(img_data)
	rd_flip_up_down=tf.image.random_flip_up_down(img_data)
	rd_flip_left_right=tf.image.random_flip_left_right(img_data)
	#img=tf.image.random_saturation(img_data,lower=0.5,upper=1.5)
	plt.imshow(img.eval())
	plt.show()
