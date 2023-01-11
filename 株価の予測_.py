from keras.models import Sequential
from keras.layers import Dense, Activation


import numpy as np

# 入力データ
x = np.array([
    [0, 1, 2],
    [1, 3, 1],
    [3, 1, -1],
    [5, 2, 0],
    [0, 8, 0]
])

# 正解データ
y = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0]
])

# モデルのインスタンス作成
model = Sequential()

# モデルにレイヤーを追加
model.add(Dense(4, input_dim=3))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy')

model.summary()

history = model.fit(x, y, epochs=500)

model.predict(x), y

from matplotlib import pyplot as plt


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
