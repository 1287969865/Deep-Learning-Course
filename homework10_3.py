import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 

boston_housing=tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y)=boston_housing.load_data()

#数据归一化处理
x_train=(train_x-train_x.min(axis=0))/(train_x.max(axis=0)-train_x.min(axis=0))
x_test=(test_x-test_x.min(axis=0))/(test_x.max(axis=0)-test_x.min(axis=0))

#数据选择
x1_train=x_train[:,7]   
x2_train=x_train[:,12] 
y_train=train_y

x1_test=x_test[:,7]    
x2_test=x_test[:,12]  
y_test=test_y

#整理数据
X_train=tf.cast(tf.stack([x1_train,x2_train],axis=1),tf.float32)
X_test=tf.cast(tf.stack([x1_test,x2_test],axis=1),tf.float32)

Y_train=tf.constant(y_train.reshape(-1,1),tf.float32)
Y_test=tf.constant(y_test.reshape(-1,1),tf.float32)

learn_rate=0.01
iter=2000
display_step=80

np.random.seed(612)
W=tf.Variable(np.random.randn(2,1),dtype=tf.float32)

mse_train=[]
mse_test=[]

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        pred_train=tf.matmul(X_train,W)
        loss_train=0.5*tf.reduce_mean(tf.square(Y_train-pred_train))

        pred_test=tf.matmul(X_test,W)
        loss_test=0.5*tf.reduce_mean(tf.square(Y_test-pred_test))

    mse_train.append(loss_train)
    mse_test.append(loss_test)

    dL_dW=tape.gradient(loss_train,W)
    W.assign_sub(learn_rate*dL_dW)

    if i % display_step==0:
        print("i:%i,Train Loss:%f,Test Loss:%f" %(i,loss_train,loss_test))
    
plt.figure(figsize=(8,4))
plt.suptitle("RM && LSTAT———— Price")

plt.subplot(131)
plt.ylabel('MSE')
plt.plot(mse_train,color='blue',linewidth=3)
plt.plot(mse_train,color='red',linewidth=1.5)

plt.subplot(132)
plt.ylabel('Price')
plt.plot(Y_train,color='blue',marker='o',label="true_price")
plt.plot(pred_train,color='red',marker='.',label="predicrt")
plt.legend()


plt.subplot(133)
plt.ylabel('Price')
plt.plot(Y_test,color='blue',marker='o',label="true_price")
plt.plot(pred_test,color='red',marker='.',label="predicrt")
plt.legend()

plt.show()
