import os
import numpy as np
import tensorflow as tf
import threading
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def MyLoop(coord,work_id):
	while not coord.should_stop():
		if np.random.rand()<0.02:
			print ("stop from id:%d\n"%(work_id))
			coord.request_stop()
		else:
			print ("work on id:%d\n"%(work_id))
		time.sleep(1)
coord=tf.train.Coordinator()
threads=[threading.Thread(target=MyLoop,args=(coord,i)) for i in xrange(5)]
for t in threads: t.start()
coord.join(threads)
