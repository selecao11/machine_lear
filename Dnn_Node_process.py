import math

# 取りあえず仮で、空の関数を定義して、コードが実行できるようにしておく
def summation(x,weights, bias):
# ※1データ分、つまりxとweightsは「一次元リスト」という前提。
    linear_sum = 0.0
    for x_i, w_i in zip(x, weights):
        linear_sum += x_i * w_i  # iは「番号」（数学は基本的に1スタート）
    linear_sum += bias
    return linear_sum

def sum_der(x, weights, bias, with_respect_to='w'):
    # ※1データ分、つまりxとweightsは「一次元リスト」という前提。
    if with_respect_to == 'w':
        # 線形和uを各重みw_iで偏微分するとx_iになる（iはノード番号）
        # 合計値の線形和uについて各重みw_iに対しての値を求めるとx_iになる
        return x  
    elif with_respect_to == 'b':
        # 線形和uをバイアスbで偏微分すると1になる
        # 合計値の線形和uについてバイアスbに対しての値を求めると1になる
        return 1.0  
    elif with_respect_to == 'x':
        # 線形和uを各入力x_iで偏微分するとw_iになる
        # 合計値の線形和uについて各入力x_iに対しての値を求めるとw_iになる
        return weights  # 線形和uを各入力x_iで偏微分するとw_iになる

def sigmoid(x):
    " シグモイド関数。"
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_der(x):
    output = sigmoid(x)
    return output * (1.0 - output)

def identity(x):
    " 恒等関数。"
    return x

def identity_der(x):
    return 1.0

def back_prop(layers, weights, biases, y_true, cached_outs, cached_sums):
    " 逆伝播を行う関数。"
    return None, None

def update_params(layers, weights, biases, grads_w, grads_b, lr=0.1):
    " パラメーター（重みとバイアス）を更新する関数。"
    return None, None

def sseloss(y_pred, y_true):
    return 0.5 * (y_pred - y_true) ** 2

def sseloss_der(y_pred, y_true):
    return y_pred - y_true


w = [0.0, 0.0]  # 重み（仮の値）
b = 0.0  # バイアス（仮の値）
x = [2,3,5]

next_x = x  # 訓練データをノードへの入力に使う

# ---ここまでは仮の実装。ここからが必要な実装---

# 1つのノードの処理（1）： 重み付き線形和
node_sum = summation(next_x, w, b)

# 1つのノードの処理（2）： 活性化関数
is_hidden_layer = True
if is_hidden_layer:
    # 隠れ層（シグモイド関数）
    node_out = sigmoid(node_sum)
else:
    # 出力層（恒等関数）
    node_out = identity(node_sum)