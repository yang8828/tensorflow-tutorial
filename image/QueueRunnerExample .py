import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

q=tf.FIFOQueue(100,'float32')
init=q.enqueue([tf.random_normal([1])])
qr=tf.train.QueueRunner(q,[init]*5)
tf.train.add_queue_runner(qr)
out_tensor=q.dequeue()
with tf.Session() as sess:
	coord=tf.train.Coordinator()
	threads=tf.train.start_queue_runners(sess=sess,coord=coord)
	for i in range(5):	
		print sess.run(out_tensor)
	
	coord.request_stop()
	coord.join(threads)
	
