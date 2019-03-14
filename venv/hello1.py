import tensorflow as tf
import numpy as np

coefficients1 = np.array([[1.], [-10.], [25.]])
coefficients2 = np.array([[1.], [-20.], [100.]])

w = tf.Variable(0, dtype = tf.float32)
x = tf.placeholder(tf.float32, [3,1])
cost = x[0][0] * w **2 + x[1][0] * w +x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

print(session.run(w))


for i in range(10000):
    session.run(train, feed_dict={x:coefficients2})

print(session.run(w))