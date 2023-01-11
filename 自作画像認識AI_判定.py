#③ 自前画像で判定（○×判定等）

#1 ライブラリのインポート等
import cv2 
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
#from keras.models import load_model #model = load_model('cnn_model_weight.h5') #model.save('cnn_model_weight.h5') で保存したファイルを読み込む場合に使用

#2 各種設定
recognise_image = '/home/user/ダウンロード/iori_正面.JPG' #ここを変更。画像認識したい画像ファイル名。（実行前に認識したい画像ファイルを1つアップロードしてください）
folder = ['正面', 'その他']  #ここを変更。今回は日本語の表示にしたかったので、folder = ['circle', 'cross'] の順番で日本語にしています。
image_size = 200 # ここを変更。「28」を指定した場合、縦28横28ピクセルの画像に変換
                # 「② 用意した自前画像で学習」と同じにする。
color_setting = 1  # ここを変更。画像認識する画像のカラー。「1」はモノクロ・グレースケール。「3」はカラー
                   # 「② 用意した自前画像で学習」と同じにする。

#3 各種読み込み
model = model_from_json(open('cnn_model.json', 'r').read())
model.load_weights('cnn_weights.h5')
#model = load_model('cnn_model_weight.h5') #model.save('cnn_model_weight.h5') で保存したファイルを読み込む場合に使用

#4 画像の表示・各種設定等
img = cv2.imread(recognise_image, 0)  #ここを変更。モノクロ・グレースケールの場合は「0」。カラーの場合は「1」 。         
img = cv2.resize(img, (image_size, image_size))
plt.imshow(img)
plt.gray()  #ここを変更。カラーの場合は「plt.gray()」を消す。モノクロ・グレースケールの場合は「plt.gray()」が無いと変な色になります。
plt.show()
img = img.reshape(image_size, image_size, color_setting).astype('float32')/255 

#5 予測と結果の表示等
prediction = model.predict(np.array([img]), batch_size=2, verbose=1)
result = prediction[0]

for i, accuracy in enumerate(result):
  print('画像認識AIは「', folder[i], '」の確率を', int(accuracy * 100), '% と予測しました。')

print('-------------------------------------------------------')
print('画像認識AI：○ or × の予測結果は、「', folder[result.argmax()],'」です。')