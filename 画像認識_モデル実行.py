import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils  import np_utils
import sys
import numpy as np
from PIL import Image
from tensorflow import keras

# クラスラベル
labels = ["grape","apple","orange"]
# ディレクトリ
dataset_dir = "data/dataset.npy" # 前処理済みデータ
model_dir   = "data/cnn_h5"      # 学習済みモデル
# リサイズ設定
resize_settings = (50,50)

# 保存したnumpyデータ読み込み
X_train,X_test,y_train,y_test = np.load(dataset_dir, allow_pickle=True)

# 推論用モデル
def predict():
    
    #インスタンス
    model = Sequential()
    # 1層目 (畳み込み）
    model.add(Conv2D(32,(3,3),padding="same", input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    # 2層目（Max Pooling)
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    # 3層目 (Max Pooling)
    model.add(MaxPooling2D(pool_size=(2,2)))                     
    model.add(Dropout(0.3))                     
    # 4層目 (畳み込み)
    model.add(Conv2D(64,(3,3),padding="same"))                   
    model.add(Activation('relu'))
    # 5層目 (畳み込み)
    model.add(Conv2D(64,(3,3))) 
    model.add(Activation('relu'))
    # 6層目 (Max Pooling)
    model.add(MaxPooling2D(pool_size=(2,2)))
    # データを1列に並べる
    model.add(Flatten())
    # 7層目 (全結合層)
    model.add(Dense(512))                                       
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # 出力層(softmaxで確率を渡す：当てはまるものを1で返す)
    model.add(Dense(3)) 
    model.add(Activation('softmax'))
    # 最適化の手法
    opt = tensorflow.keras.optimizers.RMSprop(lr=0.005, decay=1e-6)
    # 損失関数
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"]
                 )
                  

    # モデル学習(推論では不要のためコメントアウト)
    # model.fit(X_train,y_train,batch_size=10,epochs=150) 
    
    # モデルを読み込み
    model = keras.models.load_model("data/cnn_h5")
    
    return model


# 実行関数
def main(path):
    X     = []                               # 推論データ格納
    image = Image.open(path)                 # 画像読み込み
    image = image.convert("RGB")             # RGB変換
    image = image.resize(resize_settings)    # リサイズ
    data  = np.asarray(image)                # 数値の配列変換
    X.append(data)
    X     = np.array(X)
    
    # モデル呼び出し
    model = predict()
    
    # numpy形式のデータXを与えて予測値を得る
    model_output = model.predict([X])[0]
    # 推定値 argmax()を指定しmodel_outputの配列にある推定値が一番高いインデックスを渡す
    predicted = model_output.argmax()
    # アウトプット正答率
    accuracy = int(model_output[predicted] *100)
    print("{0} ({1} %)".format(labels[predicted],accuracy))