from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import cv2
import imutils

path='/home/yangqr8828/Desktop/Test_DL/data/etl_image/'

w=100
h=100
c=3

def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            img=io.imread(im)
            img_tras=transform.resize(img,(w,h))		
            imgs.append(img_tras)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data,label=read_img(path)

num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')


conv1=tf.layers.conv2d(inputs=x,  filters=32,  kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

conv2=tf.layers.conv2d( inputs=pool1, filters=64,  kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

conv3=tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

conv4=tf.layers.conv2d(  inputs=pool3, filters=128, kernel_size=[3, 3], padding="same",activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])


dense1 = tf.layers.dense(inputs=re1,      units=1024,   activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2= tf.layers.dense(inputs=dense1,  units=512,  activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits= tf.layers.dense(inputs=dense2,  units=5,   activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


n_epoch=1000
batch_size=64
with tf.Session() as sess: 
	sess.run(tf.global_variables_initializer())
	for epoch in range(n_epoch):
    		#training
    		train_loss, train_acc, n_batch = 0, 0, 0
    		for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        		_,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        		train_loss += err
			train_acc += ac	
			n_batch += 1
		print("   step: %d" % (epoch))
    		print("   train acc: %f" % (train_acc/ n_batch))
    
    		#validation
		val_loss, val_acc, n_batch = 0, 0, 0
		for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        		err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        		val_loss += err
 			val_acc += ac
			n_batch += 1
		print("   step: %d" % (epoch))
    		print("   validation acc: %f" % (val_acc/ n_batch))

