import Dnn_Forward_Prop
import Dnn_Back_Prop 

x = [0.05, 0.1]
layers = [2, 2, 2]
weights = [
    [   
        [0.15, 0.2], 
        [0.25, 0.3]
    ],
    [
        [0.4, 0.45],
        [0.5,0.55]
    ]
]
biases = [[0.35, 0.35], [0.6, 0.6]]
model = (layers, weights, biases)
y_true = [0.01, 0.99]

# （1）順伝播の実行例
y_pred, cached_outs, cached_sums = Dnn_Forward_Prop.forward_prop(*model, x, cache_mode=True)
print(f'y_pred={y_pred}')
print('\n')
print(f'cached_outs={cached_outs}')
print('\n')
print(f'cached_sums={cached_sums}')
print('\n')
grads_w, grads_b = Dnn_Back_Prop.back_prop(*model, y_true, cached_outs, cached_sums)
print(f'grads_w={grads_w}')
print('\n')
print(f'grads_b={grads_b}')


# 異なるDNNアーキテクチャーを定義してみる
""" layers2 = [
    2,  # 入力層の入力（特徴量）の数
    3,  # 隠れ層1のノード（ニューロン）の数
    2,  # 隠れ層2のノード（ニューロン）の数
    1]  # 出力層のノードの数

# 重みとバイアスの初期値
weights2 = [
        [[-0.2, 0.4], [-0.4, -0.5], [-0.4, -0.5]], # 入力層→隠れ層1
        [[-0.2, 0.4, 0.9], [-0.4, -0.5, -0.2]], # 隠れ層1→隠れ層2
        [[-0.5, 1.0]] # 隠れ層2→出力層
    ]
biases2 = [
    [0.1, -0.1, 0.1],  # 隠れ層1
    [0.2, -0.2],  # 隠れ層2
    [0.3]  # 出力層
]
 """
# モデルを定義
#model2 = (layers2, weights2, biases2)

# 仮の訓練データ（1件分）を準備
x2 = [2.3, 1.5]  # x_1とx_2の2つの特徴量

# 予測時の（1）順伝播の実行例
#y_pred = Dnn_Forward_Prop.forward_prop(*model2, x2)
#print(y_pred)  # 予測値
# 出力例：
# [0.3828840428423274]