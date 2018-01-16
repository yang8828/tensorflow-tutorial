from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

INPUT_NODE=28*28
OUTPUT_NODE=10
LAYER1_NODE=100
def get_weight_variable(shape,regularizer):
	weight=tf.get_variable('weight',shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
	if regularizer!=None:
		tf.add_to_collection('losses',regularizer(weight))
	return weight

def inference(input_tensor,regularizer):
	with tf.variable_scope('layer1'):
		weight=get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
		bias=tf.get_variable('bias',[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
		layer1=tf.nn.relu(tf.matmul(input_tensor,weight)+bias)
	with tf.variable_scope('layer2'):
		weight=get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
		bias=tf.get_variable('bias',[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
		layer2=tf.nn.relu(tf.matmul(layer1,weight)+bias)
	return layer2
