#coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 只显示 Error
import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(shape=shape,value=0.1) #disorder the parameters
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x=tf.placeholder('float',shape=[None,784])
y_=tf.placeholder('float',shape=[None,10])
x_image=tf.reshape(x,[-1,28,28,1])
#第一层卷积：
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
#第二层卷积
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
#全连接层
W_fc3=weight_variable([7*7*64,1024])
b_fc3=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc3=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc3)+b_fc3)
#Dropout
keep_prob=tf.placeholder('float')
h_fc3_drop=tf.nn.dropout(h_fc3,keep_prob)
#softmax
W_fc4=weight_variable([1024,10])
b_fc4=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc3_drop,W_fc4)+b_fc4)


cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
learning_rate=tf.placeholder('float')
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction=tf.equal(tf.arg_max(y_,1),tf.arg_max(y_conv,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        batch_image,batch_label=mnist.train.next_batch(50)
        sess.run(train_step,feed_dict={x:batch_image,
                                       y_:batch_label,
                                       keep_prob:0.5,
                                       learning_rate:1e-4})
        if i%10==0:
            print 'step%d:accuracy:%f'%(i,sess.run(accuracy,feed_dict={x:batch_image,
                                                                   y_:batch_label,
                                                                   keep_prob:0.5}))
    print sess.run(accuracy,feed_dict={x:mnist.test.images,
                                       y_:mnist.test.labels,
                                       keep_prob:0.5})




# sess=tf.Session()
# W=tf.Variable(tf.zeros([784,10]))
# b=tf.Variable(tf.zeros([10]))
# sess.run(tf.global_variables_initializer())
#
# y=tf.nn.softmax(tf.matmul(x,W)+b)
# cross_entropy=-tf.reduce_sum(y_*tf.log(y))
# train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# for i in range(1000):
#     batch=mnist.train.next_batch(50)
#     # train_step.run(feed_dict={x:batch[0],y_:batch[1]}) #or can use following code
#     sess.run(train_step,feed_dict={x:batch[0],y_:batch[1]})
# correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
# accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
# print accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})
# # print sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
# sess.close()