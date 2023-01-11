import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils  import np_utils
import numpy as np

# クラスラベル
labels = ["grape","apple","orange"]
# ディレクトリ
dataset_dir = "data/dataset.npy" # 前処理済みデータ
model_dir   = "data/cnn_h5"      # 学習済みモデル
# リサイズ設定
resize_settings = (50,50)

# メインの関数定義
def main():
    """
    ①データの前処理(エンコーディング)
    """
    # 保存したnumpyデータ読み込み
    X_train,X_test,y_train,y_test = np.load(dataset_dir, allow_pickle=True)
    
    # 0~255の整数範囲になっているため、0~1間に数値が収まるよう正規化
    X_train = X_train.astype("float") / X_train.max()
    X_test  = X_test.astype("float") /  X_train.max()
    
    # クラスラベルの正解値は1、他は0になるようワンホット表現を適用
    y_train = np_utils.to_categorical(y_train,len(labels))
    y_test  = np_utils.to_categorical(y_test,len(labels))
    """
    ②モデル学習&評価
    """
    #モデル学習
    model = model_train(X_train,y_train)
    
    #モデル評価
    evaluate(model,X_test, y_test)
    
  
    
#モデル学習関数
def model_train(X_train,y_train):
    
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
    # データを1次元化
    model.add(Flatten())
    # 7層目 (全結合層)
    model.add(Dense(512))                                       
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # 出力層(softmaxで0〜1の確率を返す)
    model.add(Dense(3)) 
    model.add(Activation('softmax'))
    # 最適化アルゴリズム
    opt = tensorflow.keras.optimizers.RMSprop(lr=0.005, decay=1e-6)
    # 損失関数
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"]
                 )
                  
    # モデル学習
    model.fit(X_train,y_train,batch_size=10,epochs=150)
    # モデルの結果を保存
    model.save(model_dir)
    return model
    

# 評価用関数
def evaluate(model,X_test,y_test):
    # モデル評価
    scores = model.evaluate(X_test,y_test,verbose=1)
    print("Test Loss: ", scores[0])
    print("test Accuracy: ", scores[1])