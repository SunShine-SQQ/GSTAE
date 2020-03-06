# -*- coding: utf-8 -*-
"""
@author: SunShine
修改日志：将真实案例改为数值案例
"""

import scipy.io as sio
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
import random as rd
import matplotlib.pyplot as plt
import math


def get_weight(shape, lamb=0.0001,  *collection):
    var = tf.Variable(tf.random_normal(shape))
    for coll in collection:
        tf.add_to_collection(coll, \
                         tf.contrib.layers.l2_regularizer(lamb)(var))
    return var

def reconstructed_loss(original, reconstruction):
    '''
    均方误差作为重建损失
    '''
    rec_loss = tf.reduce_mean(tf.pow(original - reconstruction, 2))
    return rec_loss

def predicted_loss(labels, logits):
    '以RMSE作为预测损失'
    pred_loss = tf.sqrt(tf.reduce_mean(tf.square(labels - logits)), name = 'RMSE') 
    return pred_loss



np.random.seed(42)
#读取数据
print('读取数据...')
data = np.loadtxt(open(".\data.csv","rb"),delimiter=",",skiprows=0)
#plt.figure()
#plt.plot(data[:,13])
#plt.show()


X_set = data[:,:13]
Y_set = data[:,13].reshape(-1,1)
#x_train, x_test, y_train, y_test = X_set[:1000,:], X_set[1000:,:], Y_set[:1000,:], Y_set[1000:,:]
x_train,x_test,y_train,y_test = train_test_split(X_set, Y_set, test_size=2/3, random_state=42)
scaler_X = preprocessing.StandardScaler().fit(x_train)
#scaler_Y = preprocessing.StandardScaler().fit(Y_set)
#scaler_mean = scaler_Y.mean_
#scaler_std = scaler_Y.scale_
x_trn_scaled = scaler_X.transform(x_train)
x_tst_scaled = scaler_X.transform(x_test)
print('数据读取完成，训练集大小：%d, 测试集大小： %d' % (len(y_train), len(y_test)))


#搭建模型
print('搭建模型...')
tf.reset_default_graph()
lamda = 0.5
learning_rate = 0.005

#结构参数
n_input = 13 
n_h1 = 10
n_h2 = 7
n_h3 = 4
n_output = 1

#占位符
#第一个AE
l1x = tf.placeholder('float', [None, n_input])
l1y = tf.placeholder('float', [None, n_output])
#第二个AE
l2x = tf.placeholder('float', [None, n_h1])
l2y = tf.placeholder('float', [None, n_output])
#第三个AE
l3x = tf.placeholder('float', [None, n_h2])
l3y = tf.placeholder('float', [None, n_output])
#增强特征输入
h3 = tf.placeholder('float', [None, n_h3])

#输出层
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_output])
#def AE(x,y,hidden_nodes,training_epoch=100,batch_size=128):
    
#网络参数
weights = {
        #第一个AE    
        'w1_enc': get_weight([n_input, n_h1], 0.01, 'cost_AE1'),            
        'w1_dec': 
#            tf.Variable(tf.random_normal([n_h1, n_input])), 
            get_weight([n_h1, n_input], 0.01, 'cost_AE1'),
        'w1_out':  
#            tf.Variable(tf.random_normal([n_h1, n_output])),
            get_weight([n_h1, n_output], 0.01, 'cost_AE1'),
        #第二个AE    
        'w2_enc': get_weight([n_h1, n_h2], 0.01, 'cost_AE2'),
        'w2_dec': 
#            tf.Variable(tf.random_normal([n_h2, n_h1])),
            get_weight([n_h2, n_h1], 0.01, 'cost_AE2'),
        'w2_out': 
#            tf.Variable(tf.random_normal([n_h2, n_output])),
            get_weight([n_h2, n_output], 0.01, 'cost_AE2'),
        #第三个AE    
        'w3_enc': get_weight([n_h2, n_h3], 0.01, 'cost_AE3'),
        'w3_dec': 
#            tf.Variable(tf.random_normal([n_h3, n_h2])),
            get_weight([n_h3, n_h2], 0.01, 'cost_AE3'),
        'w3_out': 
#            tf.Variable(tf.random_normal([n_h3, n_output])),
            get_weight([n_h3, n_output], 0.01, 'cost_AE3'),
        #门控输出
        'w_g1': get_weight([n_h1, n_output], 0.05, 'cost_final'),
        'w_h1': get_weight([n_h1, n_output], 0.05, 'cost_final'),
        'w_g2': get_weight([n_h2, n_output], 0.05, 'cost_final'),
        'w_h2': get_weight([n_h2, n_output], 0.05, 'cost_final'),
        'w_g3': get_weight([n_h3, n_output], 0.05, 'cost_final'),
        'w_h3': get_weight([n_h3, n_output], 0.05, 'cost_final'),
#        'w_pred': tf.Variable(tf.random_normal([n_h3, n_outkuai put])),

        } 
   

biases = {
        #第一个AE    
        'b1_enc': tf.Variable(tf.zeros([n_h1])),
        'b1_dec': tf.Variable(tf.zeros([n_input])),
        'b1_out': tf.Variable(tf.zeros([n_output])),
        #第二个AE    
        'b2_enc': tf.Variable(tf.zeros([n_h2])),
        'b2_dec': tf.Variable(tf.zeros([n_h1])),
        'b2_out': tf.Variable(tf.zeros([n_output])),
        #第三个AE    
        'b3_enc': tf.Variable(tf.zeros([n_h3])),
        'b3_dec': tf.Variable(tf.zeros([n_h2])),
        'b3_out': tf.Variable(tf.zeros([n_output])),
        #门控输出
        'b_g1': tf.Variable(tf.zeros([n_output])),
        'b_h1': tf.Variable(tf.zeros([n_output])),
        'b_g2': tf.Variable(tf.zeros([n_output])),
        'b_h2': tf.Variable(tf.zeros([n_output])),
        'b_g3': tf.Variable(tf.zeros([n_output])),
        'b_h3': tf.Variable(tf.zeros([n_output])),
#        'b_pred': tf.Variable(tf.zeros([n_output])),

        }



#预训练
##第一层网络
h1_enc = tf.nn.tanh(tf.matmul(l1x, weights['w1_enc']) + biases['b1_enc'])
h1_dec = tf.nn.tanh(tf.matmul(h1_enc, weights['w1_dec']) + biases['b1_dec'])
h1_out = tf.nn.tanh(tf.matmul(h1_enc, weights['w1_out']) + biases['b1_out'])
rp_cost_AE1 = reconstructed_loss(l1x, h1_dec) + lamda*predicted_loss(l1y, h1_out)
tf.add_to_collection('cost_AE1', rp_cost_AE1)
cost_AE1 = tf.add_n(tf.get_collection('cost_AE1'))
opti_AE1 = tf.train.AdamOptimizer(learning_rate).minimize(cost_AE1)

##第二层网络
h2_enc = tf.nn.tanh(tf.matmul(l2x, weights['w2_enc']) + biases['b2_enc'])
h2_dec = tf.nn.tanh(tf.matmul(h2_enc, weights['w2_dec']) + biases['b2_dec'])
h2_out = tf.nn.tanh(tf.matmul(h2_enc, weights['w2_out']) + biases['b2_out'])
rp_cost_AE2 = reconstructed_loss(l2x, h2_dec) + lamda*predicted_loss(l2y, h2_out)
tf.add_to_collection('cost_AE2', rp_cost_AE2)
cost_AE2 = tf.add_n(tf.get_collection('cost_AE2'))
opti_AE2 = tf.train.AdamOptimizer(learning_rate).minimize(cost_AE2)

##第三层网络
h3_enc = tf.nn.tanh(tf.matmul(l3x, weights['w3_enc']) + biases['b3_enc'])
h3_dec = tf.nn.tanh(tf.matmul(h3_enc, weights['w3_dec']) + biases['b3_dec'])
h3_out = tf.nn.tanh(tf.matmul(h3_enc, weights['w3_out']) + biases['b3_out'])
rp_cost_AE3 = reconstructed_loss(l3x, h3_dec) + lamda*predicted_loss(l3y, h3_out)
tf.add_to_collection('cost_AE3', rp_cost_AE3)
cost_AE3 = tf.add_n(tf.get_collection('cost_AE3'))
opti_AE3 = tf.train.AdamOptimizer(learning_rate).minimize(cost_AE3)

##门控
h1 = tf.nn.tanh(tf.matmul(x, weights['w1_enc']) + biases['b1_enc'])
h2 = tf.nn.tanh(tf.matmul(h1, weights['w2_enc']) + biases['b2_enc'])
h3 = tf.nn.tanh(tf.matmul(h2, weights['w3_enc']) + biases['b3_enc'])
g1 = tf.nn.sigmoid(tf.matmul(h1, weights['w_g1']) + biases['b_g1'])
g2 = tf.nn.sigmoid(tf.matmul(h2, weights['w_g2']) + biases['b_g2'])
g3 = tf.nn.sigmoid(tf.matmul(h3, weights['w_g3']) + biases['b_g3'])
y1 = tf.nn.tanh(tf.matmul(h1, weights['w_h1']) + biases['b_h1'])
y2 = tf.nn.tanh(tf.matmul(h2, weights['w_h2']) + biases['b_h2'])
y3 = tf.nn.tanh(tf.matmul(h3, weights['w_h3']) + biases['b_h3'])
y_pred = tf.multiply(g1, y1) + tf.multiply(g2, y2) + tf.multiply(g3, y3)
#a = tf.nn.sigmoid(tf.matmul(h12, weights['w_a']) + biases['b_a'])
#ha = tf.nn.tanh(tf.matmul(h12, weights['w_tanh1']) + biases['b_tanh1']) 
#o = tf.nn.sigmoid(tf.matmul(h12, weights['w_o']) + biases['b_o'])
#h = tf.multiply(f, h3) + tf.multiply(a, ha)
#y_tanh = tf.nn.tanh(tf.matmul(h, weights['w_tanh2']) + biases['b_tanh2']) 
#y_pred = tf.multiply(o, y_tanh)

##h_drop = tf.nn.dropout(h_aug, keep_prob)
##y_pred = tf.nn.tanh(tf.matmul(h_drop, weights['w_pred']) + biases['b_pred'])
#y_pred = tf.nn.tanh(tf.matmul(h, weights['w_pred']) + biases['b_pred'])
#y_pred = tf.nn.tanh(tf.matmul(x, weights['w_output']) + biases['b_output'])
pred_loss = predicted_loss(y, y_pred)
tf.add_to_collection('cost_final', pred_loss)
cost_final = tf.add_n(tf.get_collection('cost_final'))
opti_final = tf.train.AdamOptimizer(0.001).minimize(cost_final)

init = tf.global_variables_initializer()
epochs = 100
batch_size = 3
with tf.Session() as sess:
    sess.run(init)    
    ###########################################################################
    print('开始训练第一个AE...')
    avg_cost1 = []
    for epoch in range(epochs):        
        num_batch = math.ceil(len(x_trn_scaled)/batch_size)
        total_cost1 = 0.        
        batch_index = rd.sample(range(len(x_trn_scaled)), len(x_trn_scaled))
        for i in range(num_batch):            
            batch_x1 = x_trn_scaled[batch_index[i*batch_size:min((i+1)*batch_size,\
                                                                 len(batch_index))]]
            batch_y1 = y_train[batch_index[i*batch_size:min((i+1)*batch_size,\
                                                            len(batch_index))]]
            feed1 = {l1x: batch_x1, l1y: batch_y1}
            sess.run(opti_AE1, feed_dict = feed1)
            total_cost1 += sess.run(cost_AE1, feed_dict = feed1)
        avg_cost1.append(total_cost1/num_batch)
        if (epoch + 1) % 5 == 0:
            print('epoch: %02d/%02d, average cost: %.6f' \
                  % (epoch+1, epochs, total_cost1/num_batch))
    print('第一个AE训练完成')
    fig = plt.figure(1)
    plt.plot(avg_cost1)
    plt.title('avg_cost1')
    plt.show()
    print('第一次特征编码...')
    h1_train = sess.run(h1_enc, feed_dict = {l1x: x_trn_scaled})
    h1_test = sess.run(h1_enc, feed_dict = {l1x: x_tst_scaled})
    print('第一次特征编码完成，h1_train.shape = (%d, %d), h1_test.shape = (%d, %d)'\
          % (h1_train.shape[0], h1_train.shape[1], h1_test.shape[0], h1_test.shape[1]))
    ###########################################################################
    print('开始训练第二个AE...')
    avg_cost2 = []
    for epoch in range(epochs):        
        num_batch = math.ceil(len(y_train)/batch_size)
        total_cost2 = 0.        
        batch_index = rd.sample(range(len(y_train)), len(y_train))
        for i in range(num_batch):            
            batch_x2 = h1_train[batch_index[i*batch_size:min((i+1)*batch_size,\
                                                       len(batch_index))]]
            batch_y2 = y_train[batch_index[i*batch_size:min((i+1)*batch_size,\
                                                            len(batch_index))]]
            feed2 = {l2x: batch_x2, l2y: batch_y2}
            sess.run(opti_AE2, feed_dict = feed2)
            total_cost2 += sess.run(cost_AE2, feed_dict = feed2)
        avg_cost2.append(total_cost2/num_batch)
        if (epoch + 1) % 5 == 0:
            print('epoch: %02d/%02d, average cost: %.6f' \
                  % (epoch+1, epochs, total_cost2/num_batch))
    print('第二个AE训练完成')
    fig = plt.figure(2)
    plt.plot(avg_cost2)
    plt.title('avg_cost2')
    plt.show()
    print('第二次特征编码...')
    h2_train = sess.run(h2_enc, feed_dict = {l2x: h1_train})
    h2_test = sess.run(h2_enc, feed_dict = {l2x: h1_test})
    print('第二次特征编码完成，h2_train.shape = (%d, %d), h2_test.shape = (%d, %d)'\
          % (h2_train.shape[0], h2_train.shape[1], h2_test.shape[0], h2_test.shape[1]))
    ###########################################################################
    print('开始训练第三个AE...')
    avg_cost3 = []
    for epoch in range(epochs):        
        num_batch = math.ceil(len(y_train)/batch_size)
        total_cost3 = 0.
        batch_index = rd.sample(range(len(y_train)), len(y_train))
        for i in range(num_batch):            
            batch_x3 = h2_train[batch_index[i*batch_size:min((i+1)*batch_size,\
                                                       len(batch_index))]]
            batch_y3 = y_train[batch_index[i*batch_size:min((i+1)*batch_size,\
                                                            len(batch_index))]]
            feed3 = {l3x: batch_x3, l3y: batch_y3}
            sess.run(opti_AE3, feed_dict = feed3)
            total_cost3 += sess.run(cost_AE3, feed_dict = feed3)
        avg_cost3.append(total_cost3/num_batch)
        if (epoch + 1) % 5 == 0:
            print('epoch: %02d/%02d, average cost: %.6f' \
                  % (epoch+1, epochs, total_cost3/num_batch))
    print('第三个AE训练完成')
    fig = plt.figure(3)
    plt.plot(avg_cost3)
    plt.title('avg_cost3')
    plt.show()    
    print('第三次特征编码...')
    h3_train = sess.run(h3_enc, feed_dict = {l3x: h2_train})
    h3_test = sess.run(h3_enc, feed_dict = {l3x: h2_test})
    print('第三次特征编码完成，h3_train.shape = (%d, %d), h3_test.shape = (%d, %d)'\
          % (h3_train.shape[0], h3_train.shape[1], h3_test.shape[0], h3_test.shape[1]))
    ###########################################################################
    print('门控输出...')
#    h12_train = np.concatenate((h1_train, h2_train), axis=1)  
#    h12_valid_test = np.concatenate((h1_test, h2_test), axis=1) 
#    h_train = h3_train
#    h_valid_test = h3_test
    x_valid, x_test, y_valid, y_test = train_test_split(x_tst_scaled, y_test, \
                                                        test_size = 0.5, random_state = 42)
       #    print('原始特征维度：%d，新的特征维度：%d' % (n_input, h_train.shape[1]))
    print('开始训练输出模型...')
    train_rmse = []
    valid_rmse = []
    for epoch in range(150):        
        num_batch = math.ceil(len(y_train)/batch_size)
        total_rmse_trn = 0.  #训练
        total_rmse_vld = 0.  #测试
        batch_index = rd.sample(range(len(y_train)), len(y_train))
        for i in range(num_batch):            
            batch_x = x_trn_scaled[batch_index[i*batch_size:min((i+1)*batch_size,\
                                                       len(batch_index))]]
#            batch_h12 = h12_train[batch_index[i*batch_size:min((i+1)*batch_size,\
#                                                       len(batch_index))]]
            batch_y = y_train[batch_index[i*batch_size:min((i+1)*batch_size,\
                                                            len(batch_index))]]
            train_feed = {x: batch_x,  y: batch_y}            
            valid_feed = {x: x_valid,  y: y_valid}
#            train_feed = {x: batch_x, y: batch_y, keep_prob: 0.95}            
#            valid_feed = {x: h_valid, y: y_valid, keep_prob: 1.0}
            sess.run(opti_final, feed_dict = train_feed)
            total_rmse_trn += sess.run(pred_loss, feed_dict = train_feed)
            total_rmse_vld += sess.run(pred_loss, feed_dict = valid_feed)        
        train_rmse.append(total_rmse_trn/num_batch)
        valid_rmse.append(total_rmse_vld/num_batch)
        if (epoch + 1) % 5 == 0:
            print('epoch: %02d/%02d, average train cost: %.6f, average valid rmse: %.6f' \
                  % (epoch+1, 150, total_rmse_trn/num_batch, total_rmse_vld/num_batch))
#    w_aug = sess.run(weights['w_aug'])
    print('训练完成')
    fig = plt.figure(4)
    plt.plot(train_rmse,'b')
    plt.plot(valid_rmse,'r')
    plt.title('RMSE trend of train and valid dataset')
    plt.show()  
    print('测试集检验...')
#    test_feed = {x: h_test, y: y_test, keep_prob: 1.0}
    test_feed = {x: x_test, y: y_test}
    test_rmse, G1, G2, G3, Y1, Y2, Y3, y_predict = sess.run([pred_loss,\
                                                             g1, g2, g3, \
                                                             y1, y2, y3, \
                                                             y_pred], feed_dict = test_feed) 
    r2 = r2_score(y_test, y_predict)
    print('在测试集上的RMSE为：%.6f, R^2为：%.6f' % (test_rmse,r2))
    
    np.save("./test_predict_GSTAE2/rmse%f, r2=%f"% (test_rmse,r2), [y_test, y_predict])
    np.save("./test_predict_GSTAE2/gate——rmse%f, r2=%f"% (test_rmse,r2), [G1, G2, G3])
    #输出值
    fig = plt.figure(5)
    plt.plot(y_test,'-ob')
    plt.plot(y_predict,'-or')
    plt.legend(labels = ['True Value', 'Predict Value'], loc='lower right',edgecolor='black')
    plt.title('Predict performance in test dataset')
    plt.show()
    #散点图
    fig = plt.figure(6)
    plt.scatter(y_test, y_predict, marker='x')
    labelmin = min(min(y_test), min(y_predict))-0.01
    labelmax = max(max(y_test), max(y_predict))+0.01
    plt.plot([labelmin, labelmax], [labelmin, labelmax], 'k-')
    plt.xlim(labelmin, labelmax)
    plt.ylim(labelmin, labelmax)
    plt.xlabel('y_test')
    plt.ylabel('y_predict')
    plt.title('Scatter of predicton')
    plt.show()
    #误差值
    fig = plt.figure(7)
    plt.plot(y_test-y_predict,'-b')
    plt.title('Predict error in test dataset')
    plt.show()

#    ##门控值
#    fig = plt.figure(8)
#    plt.plot(G1,label= 'G1')
#    plt.plot(G2,label= 'G2')
#    plt.plot(G3,label= 'G3')
#    plt.legend()
#    plt.show()