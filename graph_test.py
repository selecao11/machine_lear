import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# -*- coding: utf-8 -*-
# データの読み込み
df = pd.read_csv("/home/user/anaconda3/envs/machine_lear/machine_lear/AGC_累積.csv", encoding="UTF-8", index_col=0)
# グラフ作成
plt.rcParams['font.family'] = "MS Gothic" # 日本語化の設定
#df.plot(figsize=(10, 6), xlim=[2000, 2014], ylim=[0, 100])
df.plot()
plt.show()
''' 
data = pd.read_csv('/home/user/anaconda3/envs/machine_lear/machine_lear/iris/iris.data.csv')
#data =pd.read_csv('/home/user/anaconda3/envs/machine_lear/machine_lear/AGC_累積.csv')
plt.figure()
data.plot()


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10) # xは0～10
y = 2 * x + 25 # y = 2x + 5

# グラフ
plt.plot(x, y)

# 表示
plt.show()
'''
