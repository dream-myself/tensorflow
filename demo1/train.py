import tensorflow as tf
import preprocess
import model
import os
batch_size = 64 #批处理大小
def train_net():

    output, X, YY, keep_prob = model.model()

    def _onehot(lables):#one-hot编码
        return tf.one_hot(lables, depth=26, on_value=1.0, axis=2)
    Y = _onehot(YY)
    print(Y)
    #损失定义
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(Y, [-1, 26 * 4]), logits= output))

    # optimizer 选择
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    predict = tf.reshape(output, [-1, 4, 26])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, 4, 26]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)

    accuracy = tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), axis=1))

    # 读取数据
    images, labels = preprocess.read_data(['deal/0.txt', 'deal/1.txt', 'deal/2.txt'])
    total, width = labels.shape

    image_test, label_test = preprocess.read_data(['deal/test.txt'])

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    #训练过程
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #训练信息写入tensorboard
        filewriter = tf.summary.FileWriter("logs/", graph=sess.graph)

        step = 0
        if os.listdir('./check_point'):
            ckpt = tf.train.latest_checkpoint('./check_point')
            # print(ckpt)
            # ckpt = './check_point/weight-3980'
            saver.restore(sess, ckpt)
            print('restore from the checkpoint: {0}'.format(ckpt))
            # images, labels = preprocess.read_data(['deal/0.txt', 'deal/1.txt', 'deal/2.txt'])
        while True:
            for i in range(int(total / batch_size)):
                start_index = (i * batch_size) % total
                image_batch = images[start_index: start_index + batch_size]
                label_batch = labels[start_index: start_index + batch_size]
                summary, _, loss_ = sess.run([merged ,optimizer, loss], feed_dict={X: image_batch, YY: label_batch, keep_prob: 0.75})
                print(step, 'loss: %f' %loss_)
                filewriter.add_summary(summary, step)
                step += 1

                if step % 10 == 0:
                    acc = sess.run(accuracy, feed_dict={X: image_test, YY: label_test, keep_prob: 1.0})
                    print('第%d步，在测试集上的准确率为 %.2f'%(step, acc))
                    if acc > 0.4:
                        saver.save(sess,'./check_point/weight', global_step= step)

if __name__ == "__main__":
    train_net()