#-*-coding:utf-8-*-#
import tensorflow as tf

def model(w_alpha = 0.01, b_alpha =0.1):

    image_batch = tf.placeholder(tf.float32, [None, 40 *100])
    label_batch = tf.placeholder(tf.int32, [None, 4])
    keep_prob = tf.placeholder(tf.float32)  # dropout
    x = tf.reshape(image_batch,shape=[-1, 40, 100, 1])
    # 3 conv layers
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    print(conv1)
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    print(conv2)
    # w_add1 = tf.Variable(w_alpha * tf.random_normal([3,3,64,64]))
    # b_add1 = tf.Variable(b_alpha * tf.random_normal([64]))
    # conv_add1  = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_add1, strides=[1, 1, 1, 1], padding='SAME'), b_add1))
    # conv_add1 = tf.nn.max_pool(conv_add1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv_add1 = tf.nn.dropout(conv_add1, keep_prob)
    #
    # w_add2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    # b_add2 = tf.Variable(b_alpha * tf.random_normal([128]))
    # conv_add2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_add1, w_add2, strides=[1, 1, 1, 1], padding='SAME'), b_add2))
    # conv_add2 = tf.nn.max_pool(conv_add2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv_add2 = tf.nn.dropout(conv_add2, keep_prob)


    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    print(conv3)
    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8320, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, 26 * 4]))
    b_out = tf.Variable(b_alpha * tf.random_normal([26 * 4]))
    out = tf.add(tf.matmul(dense, w_out), b_out)


    return out, image_batch, label_batch, keep_prob


