import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict = {input1 : [7], input2 : [2]}))


