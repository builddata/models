# -*- coding: utf-8 -*-
"""
Tensorflow

"""
#cudart64_101 放在C:\Program Files\NVIDIA Corporation\NvStreamSrv

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:/data_analysis/models/MNIST_data/MNIST_data", one_hot=True)
X_train,Y_train = mnist.train.images,mnist.train.labels
X_test,Y_test = mnist.test.images,mnist.test.labels
#placeholder 我们一般给X、Y定义成一个placeholder，即占位符。也就是在构建图的时候，
# 我们X、Y的壳子去构建，因为这个时候我们还没有数据，但是X、Y是我们图的开端，
# 所以必须找一个什么来代替。这个placeholder就是代替真实的X、Y来进行图的构建的，
# 它拥有X、Y一样的形状。 等session开启之后，我们就把真实的数据注入到这个placeholder中即可

X = tf.placeholder(dtype=tf.float32,shape=[None,784],name='X')
Y = tf.placeholder(dtype=tf.float32,shape=[None,10],name='Y')

# 定义各个参数：最好给每个tensor 都取个名字（name属性），这样报错的时候，我们可以方便地知道是哪个
W1 = tf.get_variable('W1',[784,128],initializer=tf.keras.initializers.glorot_normal(seed=1))
b1 = tf.get_variable('b1',[128],initializer=tf.zeros_initializer())
W2 = tf.get_variable('W2',[128,64],initializer=tf.keras.initializers.glorot_normal(seed=1))
b2 = tf.get_variable('b2',[64],initializer=tf.zeros_initializer())
W3 = tf.get_variable('W3',[64,10],initializer=tf.keras.initializers.glorot_normal(seed=1))
b3 = tf.get_variable('b3',[10],initializer=tf.zeros_initializer())
#我们根据上面的变量，来 计算网络中间的logits（就是我们常用的Z）、激活值：
A1 = tf.nn.relu(tf.matmul(X,W1)+b1,name='A1')
A2 = tf.nn.relu(tf.matmul(A1,W2)+b2,name='A2')
Z3 = tf.matmul(A2,W3)+b3
#为什么要用 reduce_mean()函数呢？因为经过softmax_cross_entropy_with_logits计算出来是，
# 是所有样本的cost拼成的一个向量，有m个样本，它就是m维，因此我们需要去平均值来获得一个整体的cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
#这里我们采用Adam优化器，用它来minimize cost。当然，我们可以在AdamOptimizer()中设置一些超参数，比如leaning_rate，
# 但是这里我直接采用它的默认值了，一般效果也不错
trainer = tf.train.AdamOptimizer().minimize(cost)

#接下来，我们就可以启动session，放水了！
with tf.Session() as sess:
    # 首先给所有的变量都初始化（不用管什么意思，反正是一句必须的话）：
    sess.run(tf.global_variables_initializer())
    # 定义一个costs列表，来装迭代过程中的cost，从而好画图分析模型训练进展
    costs = []
    # 指定迭代次数：
    for it in range(5000):
        # 这里我们可以使用mnist自带的一个函数train.next_batch，可以方便地取出一个个地小数据集，从而可以加快我们的训练：
        X_batch,Y_batch = mnist.train.next_batch(batch_size=64)
        # 我们最终需要的是trainer跑起来，并获得cost，所以我们run trainer和cost，同时要把X、Y给feed进去：
        _,batch_cost = sess.run([trainer,cost],feed_dict={X:X_batch,Y:Y_batch})
        costs.append(batch_cost)
        # 每100个迭代就打印一次cost：
        if it%100 == 0:
            print('iteration%d ,batch_cost: '%it,batch_cost)
    # 训练完成，我们来分别看看来训练集和测试集上的准确率：



    predictions = tf.equal(tf.argmax(tf.transpose(Z3)),tf.argmax(tf.transpose(Y)))
    accuracy = tf.reduce_mean(tf.cast(predictions,'float'))
    print("Training set accuracy: ",sess.run(accuracy,feed_dict={X:X_train,Y:Y_train}))
    print("Test set accuracy:",sess.run(accuracy,feed_dict={X:X_test,Y:Y_test}))
    z3, acc = sess.run([Z3, accuracy], feed_dict={X: X_test, Y: Y_test})
    # 随机从测试集中抽一些图片（比如第i*10+j张图片），然后取出对应的预测（即z3[i*10+j]）：
    fig, ax = plt.subplots(4, 4, figsize=(15, 15))
    fig.subplots_adjust(wspace=0.1, hspace=0.7)
    for i in range(4):
        for j in range(4):
            ax[i, j].imshow(X_test[i * 10 + j].reshape(28, 28))
            # 用argmax函数取出z3中最大的数的序号，即为预测结果：
            predicted_num = np.argmax(z3[i * 10 + j])
            # 这里不能用tf.argmax，因为所有的tf操作都是在图中，没法直接取出来
            ax[i, j].set_title('Predict:' + str(predicted_num))
            ax[i, j].axis('off')