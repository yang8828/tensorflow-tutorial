from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import tensorflow as tf
import inference
import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


BATCH_SIZE=100
LEARNING_RATE=0.8
LAERNING_RATE_DECAY=0.99
REGULARAZTION_RATE=0.0001
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH='/home/yangqr8828/Desktop/Test_DL/tensorflow/data/mnist_test/MODELS/'
MODEL_NAME='model.ckpt'

def train(mnist):
	x=tf.placeholder(tf.float32,[None,inference.INPUT_NODE],name='X')
	y_=tf.placeholder(tf.float32,[None,inference.OUTPUT_NODE],name='Y')
	regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
	y=inference.inference(x,regularizer)
	global_steps=tf.Variable(0,trainable=False)
	variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_steps)
	variables_average_ops=variable_averages.apply(tf.trainable_variables())
	cross_entrophy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cross_entrophy_mean=tf.reduce_mean(cross_entrophy)
	loss=cross_entrophy_mean+tf.add_n(tf.get_collection('losses'))
	learning_rate=tf.train.exponential_decay(LEARNING_RATE,global_steps,mnist.train.num_examples/BATCH_SIZE,LAERNING_RATE_DECAY)
	train_steps=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_steps)
	with tf.control_dependencies([train_steps,variables_average_ops]):
		train_op=tf.no_op(name='train')
	saver=tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(TRAINING_STEPS):
			xs,ys=mnist.train.next_batch(BATCH_SIZE)
			_,loss_value,step=sess.run([train_op,loss,global_steps],feed_dict={x:xs,y_:ys})
			if i%1000==0:
				print("After %d training steps, loss on training batch is %g." %(step,loss_value))
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_steps)
def main(argv=None):
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	train(mnist)
if __name__=='__main__':
	tf.app.run()
	
			
