import tensorflow as tf
import numpy as np

x_data = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
x_data = x_data.T
y_data = np.array([1, 1, 0, 0])
y_data = y_data.reshape([4,1])
print(x_data)
print(y_data)

x_place = tf.placeholder(tf.float32, [2, 4], name = "x_place")
y_place = tf.placeholder(tf.float32, [4, 1], name = "y_place")

w1 = tf.Variable(tf.random_normal([2, 2], stddev = 1, dtype = tf.float32))
b1 = tf.Variable(tf.random_normal([2, 1], stddev = 1, dtype = tf.float32))
w2 = tf.Variable(tf.random_normal([1, 2], stddev = 1, dtype = tf.float32))

a1 = tf.nn.relu(tf.matmul(w1 , x_place) + b1)
a2 = tf.matmul(w2 , a1)

cross_entropy = tf.reduce_mean(tf.square(y_place - a2))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    steps = 10000
    for i in range(steps):
        sess.run(train_step, feed_dict = {x_place:x_data, y_place:y_data})
        if i % 1000 == 0:
            cross_all = sess.run(cross_entropy, feed_dict = {x_place:x_data, y_place:y_data})
            print(i, cross_all)


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    print("w1 = ", sess.run(w1))
    print("b1 = ", sess.run(b1))
    print("w2 = ", sess.run(w2))


