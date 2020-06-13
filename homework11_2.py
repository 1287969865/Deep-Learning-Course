import tensorflow as tf
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np

#鸢尾花训练集
TRAIN_URL="http://download.tensorflow.org/data/iris_training.csv"
train_path=tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)

#鸢尾花测试集
TEST_URL="http://download.tensorflow.org/data/iris_test.csv"
test_path=tf.keras.utils.get_file(TEST_URL.split('/')[-1],TEST_URL)

#读取训练集、测试集
df_iris_train=pd.read_csv(train_path,header=0)
df_iris_test=pd.read_csv(test_path,header=0)

#转化为numpy数组
iris_train=np.array(df_iris_train)
iris_test=np.array(df_iris_test)
print("iris_train.shape:",iris_train.shape)
print("iris_test.shape:",iris_test.shape)

#分别从训练集和测试集中选择两种、三种、四种属性组合
train_x=iris_train[:,0:4]
train_y=iris_train[:,4]
test_x=iris_test[:,0:4]
test_y=iris_test[:,4]

#提取变色鸢尾标签和维吉尼亚鸢尾标签
x_train=train_x[train_y>0]
y_train=train_y[train_y>0]

x_test=test_x[test_y>0]
y_test=test_y[test_y>0]

print("x_train.shape：",x_train.shape)
print("y_train.shape：",y_train.shape)
print("x_test.shape:",x_test.shape)
print("y_test.shape：",y_test.shape)

num_train=len(x_train)
for i in range(0,num_train):
    if y_train[i]==2.0:
        y_train[i]=0.0
num_test=len(x_test)
for i in range(0,num_test):
    if y_test[i]==2.0:
        y_test[i]=0.0

#可视化样本
plt.figure(figsize=(8,4))
cm_pt=mpl.colors.ListedColormap(["red","green"])

plt.subplot(121)
plt.title("Train Data")
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_pt)
plt.subplot(122)
plt.title("Test Data")
plt.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap=cm_pt)
plt.show()

#属性按列中心化
x_train=x_train-np.mean(x_train,axis=0)
x_test=x_test-np.mean(x_test,axis=0)
plt.figure(figsize=(8,4))
cm_pt=mpl.colors.ListedColormap(["red","green"])

plt.suptitle("column centric")
plt.subplot(121)
plt.title("Train Data")
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_pt)
plt.subplot(122)
plt.title("Test Data")
plt.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap=cm_pt)
plt.show()

#生成多元模型的属性矩阵和标签列向量
x0_train=np.ones(num_train).reshape(-1,1)
X_train=tf.cast(tf.concat((x0_train,x_train),axis=1),tf.float32)
Y_train=tf.cast(tf.reshape(y_train,(-1,1)),tf.float32)
print("X_train.shape:",X_train.shape)
print("Y_train.shape:",Y_train.shape)

x0_test=np.ones(num_test).reshape(-1,1)
X_test=tf.cast(tf.concat((x0_test,x_test),axis=1),tf.float32)
Y_test=tf.cast(tf.reshape(y_test,(-1,1)),tf.float32)
print("X_test.shape:",X_test.shape)
print("Y_test.shape:",Y_test.shape)

#设置超参数
learn_rate=0.3
iter=300
display_step=30

#设置模型参数初始值
np.random.seed(612)
W=tf.Variable(np.random.randn(5,1),dtype=tf.float32)

#给出横纵坐标
x_=[-1.5,1.5]
y_=-(W[0]+W[1]*x_)/W[2]

#绘制决策边界
plt.figure(figsize=(4,4))
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_pt)
plt.plot(x_,y_,color="red",linewidth=3)
plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])

#训练模型
ce_train=[]    #保存每次迭代后的交叉熵损失
acc_train=[]   #保存准确率
ce_test=[]    #保存每次迭代后的交叉熵损失
acc_test=[]   #保存准确率

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        PRED_train=1/(1+tf.exp(-tf.matmul(X_train,W)))
        Loss_train=-tf.reduce_mean(Y_train*tf.math.log(PRED_train)+(1-Y_train)*tf.math.log(1-PRED_train))
        PRED_test=1/(1+tf.exp(-tf.matmul(X_test,W)))
        Loss_test=-tf.reduce_mean(Y_test*tf.math.log(PRED_test)+(1-Y_test)*tf.math.log(1-PRED_test))

    accuracy_train=tf.reduce_mean(tf.cast(tf.equal(tf.where(PRED_train.numpy()<0.5,0.,1.),Y_train),tf.float32))
    accuracy_test=tf.reduce_mean(tf.cast(tf.equal(tf.where(PRED_test.numpy()<0.5,0.,1.),Y_test),tf.float32))
        
    ce_train.append(Loss_train)
    ce_test.append(Loss_test)
    acc_train.append(accuracy_train)
    acc_test.append(accuracy_test)

    dL_dW=tape.gradient(Loss_train,W)
    W.assign_sub(learn_rate*dL_dW)
    
    if i%display_step==0:
        print("i:%i || TrainAcc:%f || TrainLoss:%f || TestAcc:%f || TestLoss:%f" %(i,accuracy_train,Loss_train,accuracy_test,Loss_test))
        y_=-(W[0]+W[1]*x_)/W[2]
        plt.plot(x_,y_)

#绘制损失和准确率变化曲线
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(ce_train,color="blue",label="train")
plt.plot(ce_test,color="red",label="test")
plt.ylabel("Loss")
plt.legend()

plt.subplot(122)
plt.plot(acc_train,color="blue",label="train")
plt.plot(acc_test,color="red",label="test")
plt.ylabel("Accuracy")
plt.legend()
plt.show()