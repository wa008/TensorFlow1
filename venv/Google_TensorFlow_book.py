import tensorflow as tf
from numpy.random import RandomState

batch_size = 1000

w1 = tf.Variable(tf.random_normal([4, 2], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([1, 4], stddev = 1, seed = 1))
b1 = tf.Variable(tf.random_normal([4, 1], stddev = 1, seed = 1))
b2 = tf.Variable(tf.random_normal([1, 1], stddev = 1, seed = 1))

x = tf.placeholder(tf.float32, shape = (2, None), name = "x_input")
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = "y_input")

m1 = tf.matmul(w1, x)
z1 = tf.add(m1, b1)
m2 = tf.matmul(w2, z1)
y = tf.add(m2, b2)

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128000
X = rdm.rand(2, dataset_size)
XT = X.T
Y = [[int(x1 + x2 < 1)] for (x1, x2) in XT]

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    STEPS = 10000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict = {x:X[start:end], y_:Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print("After %d training step(s), cross_entropy all data is %g" % (i,total_cross_entropy))

    print("After w1 =", sess.run(w1))
    print("After w2 =", sess.run(w2))
    print("After w3 =", sess.run(w3))

