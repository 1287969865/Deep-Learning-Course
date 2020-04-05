import tensorflow as tf
import numpy as np

area = np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
room_num = np.array([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2])
price = np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
x0 = np.ones(len(area))
X = np.stack((x0,area,room_num), axis=1)
Y = np.array(price).reshape(-1,1)

Xt = np.transpose(X)
XtX_1 = np.linalg.inv(np.matmul(Xt,X))
XtX_1_Xt = np.matmul(XtX_1,Xt)
W = np.matmul(XtX_1_Xt,Y)

W = W.reshape(-1)


x1 = float(input("请输入面积:"))
while(x1>500 or x1<20 or isinstance(x1, (int,float)) != True):
    print("输入的面积范围不合理或输入的类型错误")
    x1 = float(input("请重新输入面积:"))
x2 = int(input("请输入房间数"))
while(x2 > 10 or x1 < 1 or isinstance(x2, (int)) != True):
    print("输入的房间数范围不合理或输入的类型错误")
    x2 = int(input("请重新输入房间数:"))
y = W[1] * x1 + W[2] * x2 + W[0]
print("预测价格:" ,round(y,2),"万元")
