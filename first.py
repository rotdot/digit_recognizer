import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import batch
from tensorflow.examples.tutorials.mnist import input_data

num_pixels = 784
num_judges = 10
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

def read_train(num_judges, filename):
    image = []
    answer = []
    for i, line in enumerate(open(filename)):
        if i == 0:
            continue
        arr_line = [int(x) for x in line.split(',')]
        image.append(arr_line[1:])
        add_answer = [0 for i in range(num_judges)]
        add_answer[arr_line[0]] = 1
        answer.append(add_answer)
    return [image, answer]

def read_test(filename):
    image = []
    for i, line in enumerate(open(filename)):
        if i == 0:
            continue
        arr_line = [int(x) for x in line.split(',')]
        image.append(arr_line)
    return image

def all_divide(array):
    return [x / 252.0 for x in array]

# image, answer = map(batch.Batch, read_train(num_judges, __file__ + '/../train.csv'))
jmage, answer = read_train(num_judges, __file__ + '/../train.csv')
answer = batch.Batch(answer)
image = batch.Batch(list(map(all_divide, jmage)))
test  = list(map(all_divide, read_test(__file__ + '/../test.csv')))

x = tf.placeholder(tf.float32, [None, num_pixels])
w = tf.Variable(tf.zeros([num_pixels, num_judges]))
w0 = tf.Variable(tf.zeros([num_judges]))
f = tf.matmul(x, w) + w0
p = tf.nn.softmax(f)
t = tf.placeholder(tf.float32, [None, num_judges])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
answers = tf.argmax(p ,1)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

num = 1
batch_xs, batch_ts = mnist.train.next_batch(num)

for i in range(1, 10000):
    sess.run(train_step, feed_dict = {x: image.next_batch(100), t: answer.next_batch(100)})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict = {x: image.all_data(), t: answer.all_data()})
        print('Step: %d, Loss: %f, Acuuracy: %f' % (i, loss_val, acc_val))

# for i in range(1, 2001):
#     batch_xs, batch_ts = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict = {x: batch_xs, t: batch_ts})
#     if i % 100 == 0:
#         loss_val, acc_val = sess.run([loss, accuracy], feed_dict = {x: mnist.test.images, t: mnist.test.labels})
#         print('Step: %d, Loss: %f, Acuuracy: %f' % (i, loss_val, acc_val))
answers_val = sess.run(answers, feed_dict = {x: test})
print(answers_val)
string = 'ImageId,Label\n'
for i, answer_val in enumerate(answers_val):
    # string = string + str(i+1) +',' + string(answer_val) + '\n'
    string = '{0}{1},{2}\n'.format(string, str(i+1), str(answer_val))
f = open('text.txt', 'w') # 書き込みモードで開く
f.write(string) # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる
