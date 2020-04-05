import tensorflow as tf


a = tf.constant([ 64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])
b = tf.constant([ 62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])

numpy_x = a.numpy()
numpy_y = b.numpy()
total = tf.size(numpy_x)

s1 = 0
s2 = 0
s3 = 0
s4 = 0

for i in range(total):
    s1 += numpy_x[i] * numpy_y[i]
    s2 += numpy_x[i]
    s3 += numpy_y[i]
    s4 += pow(numpy_x[i],2)
w = (total * s1 - s2 * s3) / (total * s4 - pow(s2,2))
print("w = " + str(w.numpy()))

b = (s3 - w.numpy() * s2) / total
print("b = " + str(b.numpy()))
