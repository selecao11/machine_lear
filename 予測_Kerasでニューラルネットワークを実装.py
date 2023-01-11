import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.layers.core import Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import r2_score


# データの読み込み
df = pd.read_csv("df.csv")

# 欠損値の行の削除
# 今回は欠損値ないから不要
# df = df.dropna()

# データの分割
(train, test) = train_test_split(df, test_size=0.2, shuffle=True)

# pythonでは1行目を0行目と数える
x_train = train.iloc[:, [0, 1]]
y_train = train.iloc[:, [2]]

x_test = test.iloc[:, [0, 1]]
y_test = test.iloc[:, [2]]


#データの標準化
x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())


# モデルの構築
# inputの数:A, B
n_in = 2

# ノードの数(多いほど精度が上がるかもだけど、過学習の可能性や、処理が重いため今回は8
n_hidden = 8

# outputの数:C
n_out = 1

# 学習回数
epochs = 50
batch_size = 8

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation("relu"))
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation("relu"))
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation("relu"))
model.add(Dense(units=n_out))
model.summary()
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(loss='mean_squared_error', optimizer=optimizer)

 # 学習オプション
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

test_predict = model.predict(x_test)

# MSE
test_MSE = mean_squared_error(test_predict, y_test)
print('test MSE', test_MSE)

# 決定係数
R2 = r2_score(y_test, test_predict)
print('R2', R2)

Y_test1 = np.array(y_test)
plt.figure()
plt.scatter(Y_test1, test_predict, c='blue', label='Test', alpha=0.8)
plt.legend(loc=4)
plt.ylim(plt.ylim())
_ = plt.plot([0, 10], [0, 10])
plt.savefig('FFNN_test.pdf')
plt.show()
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='loss')
plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('FFNN_epochs.pdf')
plt.show()

#モデルの構造を保存する
model_json = model.to_json()
with open("FFNN_model.json","w") as json_file:
 json_file.write(model_json)
#モデルの重みを保存する">モデルの重みを保存する
model.save_weights("FFNN_model.h5")