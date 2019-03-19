import tensorflow as tf
import numpy as np
from numpy.random import RandomState

batch_size = 1000

w1 = tf.Variable(tf.random_normal([5, 2], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([8, 5], stddev = 1, seed = 1))
w3 = tf.Variable(tf.random_normal([1, 8], stddev = 1, seed = 1))
b1 = tf.Variable(tf.random_normal([5, 1], stddev = 1, seed = 1))
b2 = tf.Variable(tf.random_normal([8, 1], stddev = 1, seed = 1))
b3 = tf.Variable(tf.random_normal([1, 1], stddev = 1, seed = 1))

x = tf.placeholder(tf.float32, shape = (2, None), name = "x_input")
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = "y_input")

a1 = tf.nn.relu(tf.add(tf.matmul(w1, x), b1))
a2 = tf.nn.relu(tf.add(tf.matmul(w2, a1), b2))
y = tf.nn.relu(tf.add(tf.matmul(w3, a2), b3))

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128000
X = rdm.rand(2, dataset_size)
XT = X.T
Y = np.array([[int(x1 + x2 < 1)] for (x1, x2) in XT])
print("Y.shape = ", Y.shape)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        if i == 0 or i == 1:
            print("start,end,i = ",start, end, i)
            print("a1.shape =", a1.shape)
            print("a2.shape =", a2.shape)
            # print("X = ", X)
            # print("Y = ", Y)
            # print(type(X))
            # print(type(Y))
            # print("shape = ", X.shape, Y.shape)
            # mid2 = Y[start:end, :].reshape(end - start)
            # # print("mid = ",mid)
            # print("shape = ",mid2.shape)
        sess.run(train_step, feed_dict = {x:X[:, start:end], y_:Y[start:end, :]})
        if i % 500 ==0:
            print("mid work = ", i)
        if i % 1000 == 0 and i > 0:
            mid_cross_entropy = sess.run(cross_entropy, feed_dict = {x:X[:, start:end], y_:Y[start:end,:]})
            print("After %d training step(s), cross_entropy all data is %g" % (i,mid_cross_entropy))
            # print("a1.shape =", a1.shape)
            # print("a2.shape =", a2.shape)
            # total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            # print("After %d training step(s), cross_entropy all data is %g" % (i,total_cross_entropy))

    print("After w1 =", sess.run(w1))
    print("After w2 =", sess.run(w2))
    print("After w3 =", sess.run(w3))
    print("After w1.shape =", w1.shape)
    print("After w2.shape =", w2.shape)
    print("After w3.shape =", w3.shape)

