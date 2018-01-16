from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import tensorflow as tf
import cov_inference
import cov_train 
import input_data
import numpy as np
BATCH_SIZE=100
EVAL_INTERVAL_SECS=3

def evaluate(mnist):
	with tf.Graph().as_default() as g:
		x=tf.placeholder(tf.float32,[BATCH_SIZE,cov_inference.IMAGE_SIZE,cov_inference.IMAGE_SIZE,cov_inference.NUM_CHANNELS],name='x-input')
		y_=tf.placeholder(tf.float32,[None,cov_inference.OUTPUT_NODE],name='Y')
		y=cov_inference.inference(x,False,None)
		correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
		accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		variable_averages=tf.train.ExponentialMovingAverage(cov_train.MOVING_AVERAGE_DECAY)
		variables_to_restore=variable_averages.variables_to_restore()
		saver=tf.train.Saver(variables_to_restore)
		while True:
			with tf.Session() as sess:
				ckpt=tf.train.get_checkpoint_state(cov_train.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess,ckpt.model_checkpoint_path)
					global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					xs,ys=mnist.validation.next_batch(BATCH_SIZE)
					reshaped_xs=np.reshape(xs,[BATCH_SIZE,cov_inference.IMAGE_SIZE,cov_inference.IMAGE_SIZE,cov_inference.NUM_CHANNELS])
					accuracy_score=sess.run(accuracy,feed_dict={x:reshaped_xs,y_:ys})
					print("After %s training steps, validation accuracy is %g." %(global_step,accuracy_score))
				else:
					print("No CheckPoints file found")
					#return
				time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	evaluate(mnist)
if __name__=='__main__':
	tf.app.run()
	
			
