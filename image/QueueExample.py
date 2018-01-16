import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.platform import gfile
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#q=tf.FIFOQueue(2,'int32')
q=tf.RandomShuffleQueue(capacity=2, min_after_dequeue=1, dtypes="int32")
init=q.enqueue_many(([0,10],))
x=q.dequeue()
y=x+1
q_inc=q.enqueue([y])
with tf.Session() as sess:
	init.run()
	for _ in range(5):
		v,_=sess.run([x,q_inc])
		print v
