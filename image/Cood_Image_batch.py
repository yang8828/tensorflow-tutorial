import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
image_size=299
numchannels=3

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
		bbox=tf.constant([[[0,0,0.9,0.7],[0.35,0.47,0.5,0.56]]])
	if image.dtype!=tf.float32:
		image=tf.image.convert_image_dtype(image,dtype=tf.float32)
	#bbox_begin,bbox_size,_=tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
	#distorted_image=tf.slice(image,bbox_begin,bbox_size)
	distorted_image=tf.image.resize_images(image,[height,width],method=1)
	distorted_image=tf.image.random_flip_up_down(distorted_image)
	distorted_image=tf.image.random_flip_left_right(distorted_image)
	distorted_image=distort_color(distorted_image,np.random.randint(3))
	return distorted_image


#reading TFRecord
files=tf.train.match_filenames_once('/home/yangqr8828/Desktop/Test_DL/mnist_test/image/*.tfrecords')
filename_queue=tf.train.string_input_producer(files,shuffle=False)
reader=tf.TFRecordReader()
_,serialized_example=reader.read(filename_queue)
features=tf.parse_single_example(
	serialized_example,
	features={
		'image':tf.FixedLenFeature([],tf.string),
		'label':tf.FixedLenFeature([],tf.int64),
		'height':tf.FixedLenFeature([],tf.int64),
		'width':tf.FixedLenFeature([],tf.int64),
		'channels':tf.FixedLenFeature([],tf.int64),
		})
image,label=features['image'],features['label']
height,width=features['height'],features['width']
channels=features['channels']

#image preprocessing
decode_image=tf.decode_raw(image,tf.uint8)
decode_image.set_shape([image_size*image_size*numchannels])
decode_image=tf.reshape(decode_image,[image_size,image_size,numchannels])
distorted_image=preprocess_for_train(decode_image,image_size,image_size,None)

#putting images into batches for training
min_after_dequeue=10000
batch_size=100
capacity=min_after_dequeue+3*batch_size
image_batch,label_batch=tf.train.shuffle_batch([distorted_image,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)
#building neural network
logit=inference(image_batch)
loss=calc_loss(logit,label_batch)
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	coord=tf.train.Coordinator()
	threads=tf.train.start_queue_runners(sess=sess,coord=coord)
	for i in range(TRAINSTEPS):
		sess.run(train_step)
	coord.request_stop()
	coord.join(threads)
