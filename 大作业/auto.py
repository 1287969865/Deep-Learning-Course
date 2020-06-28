import numpy as np
import os
import tensorflow as tf
import kerastuner as kt

hit_dict = {0: "Pass", 1:"Hit"}

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 分数计算
def score(hand):
    new_hand = []
    for i in hand:
        if i < 11:
            new_hand.append(i)
        else:
            new_hand.append(10)
    score = sum(new_hand)
    return score

# 如果分数大于等于21，不在获得
def stillin(array):
    if score(array) < 21:
        if 1 in array and score(array) == 11:
            return False
        else:
            return True
    return False


# 创建
def build_model(hp):
    model = tf.keras.Sequential()
    hp_units = hp.Int('units', min_value=16, max_value=512, step=16)

    model.add(tf.keras.layers.Dense(hp_units, activation='relu', input_shape=(maxlen,)))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(hp_learning_rate),
                  metrics=['accuracy'])
    return model


# 输出持有的数
def print_hand(hand):
    cards = {1: 'A', 2: '2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
            10:'10', 11:'J', 12:'Q', 13:'K'}
    print("Hand: ", end = "")
    for i in hand:
        if i > 0:
            print(cards[i], end= " ")
    print("\nScore =" , score(hand), end = " ")
    if 1 in hand:
        print("or", score(hand) + 10)
    print("")
    if score(hand) > 21:
        print("Model lost!")
        return False
    return True

# 开始游戏
def play():
    hand = [np.random.randint(1,14),np.random.randint(1,14)] + [0] * (maxlen - 2)
    hit = True
    while(hit):
        if (print_hand(hand)):
            hit = np.round(model.predict([hand]))
            print("Model: " + hit_dict[hit[0][0]] + "\n")
            hand[np.count_nonzero(hand)] = np.random.randint(1,14)
        else:
            break

# 生成训练集
train = []
for i in range(1000): ## initialize hands
    hand = [np.random.randint(1, 14), np.random.randint(1, 14)]
    while stillin(hand):
        train.append(hand.copy())
        hand.append(np.random.randint(1,14))



#初始化结果集 。添加另一个数字，判断是否大于21
results = []
for hand in train:
    hit = np.random.randint(1,14)
    if score(hand) + hit > 21:
        results.append(0)
    else:
        results.append(1)

print("train长度：" + str(len(train)))
print("result长度：" + str(len(results)))


maxlen = max([len(i) for i in train])
for hand in train:
    hand += [0] * (maxlen - len(hand))


print(np.shape(train))

# 简单的带有密集层模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, input_shape = (maxlen,), activation = 'relu'))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
history = model.fit(np.array(train), np.array(results),epochs=50)
print(history)

# 利用KerasTuner找到模型最佳参数
tuner = kt.Hyperband(build_model, objective = 'accuracy',  max_epochs = 10, factor = 3)
tuner.search(train, results, epochs = 100, verbose = 2)

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(np.array(train), np.array(results),epochs=50, verbose = 2)

#测试
play()