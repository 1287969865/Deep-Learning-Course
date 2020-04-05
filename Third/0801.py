import tensorflow as tf
import numpy as np

x=[ 64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]
y=[ 62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]

a = tf.constant(x)
b = tf.constant(y)

total = tf.size(a)

avg_x = tf.reduce_mean(a)
avg_y = tf.reduce_mean(b)

s1 = 0
s2 = 0

for i in range(total):
    s1 += (a[i] - avg_x) * (b[i] - avg_y)
    s2 += pow((a[i] - avg_x),2)
w = s1 / s2
print("w = " + str(w.numpy()))

b = avg_y - w * avg_x
print("b = " + str(b.numpy()))