# ニューラルネットワークは3層構成
layers = [
    2,  # 入力層の入力（特徴量）の数
    3,  # 隠れ層1のノード（ニューロン）の数
    1]  # 出力層のノードの数

# 重みとバイアスの初期値
weights = [
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], # 入力層→隠れ層1
    [[0.0, 0.0, 0.0]] # 隠れ層1→出力層
]
biases = [
    [0.0, 0.0, 0.0],  # 隠れ層1
    [0.0]  # 出力層
]
x = [0.05, 0.1]  # x_1とx_2の2つの特徴量
w = [0.0, 0.0]  # 重み（仮の値）
b = 0.0  # バイアス（仮の値）

# モデルを定義
model = (layers, weights, biases)

import math
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_der(x):
    output = sigmoid(x)
    return output * (1.0 - output)

def sseloss(y_pred, y_true):
    return 0.5 * (y_pred - y_true) ** 2

def sseloss_der(y_pred, y_true):
    return y_pred - y_true

# 取りあえず仮で、空の関数を定義して、コードが実行できるようにしておく
def summation(x, weights, bias):
    # ※1データ分、つまりxとweightsは「一次元リスト」という前提。
    " 重み付き線形和の関数。"
    linear_sum = 0.0
    for x_i, w_i in zip(x, weights):
        linear_sum += x_i * w_i  # iは「番号」（数学は基本的に1スタート）
    linear_sum += bias
    return linear_sum

def sum_der(x, weights, bias, with_respect_to='w'):
    " シグモイド関数。"
    # ※1データ分、つまりxとweightsは「一次元リスト」という前提。
    if with_respect_to == 'w':
        return x  # 線形和uを各重みw_iで偏微分するとx_iになる（iはノード番号）
    elif with_respect_to == 'b':
        return 1.0  # 線形和uをバイアスbで偏微分すると1になる
    elif with_respect_to == 'x':
        return weights  # 線形和uを各入力x_iで偏微分するとw_iになる


def identity(x):
    " 恒等関数。"
    return x

def identity_der(x):
    return 1.0

# 取りあえず仮で、空の関数を定義して、コードが実行できるようにしておく
def forward_prop(layers, weights, biases, x, cache_mode=False):
    """
    順伝播を行う関数。
    - 引数：
    (layers, weights, biases): モデルを指定する。
    x: 入力データを指定する。
    cache_mode: 予測時はFalse、訓練時はTrueにする。これにより戻り値が変わる。
    - 戻り値：
    cache_modeがFalse時は予測値のみを返す。True時は、予測値だけでなく、
        キャッシュに記録済みの線形和（Σ）値と、活性化関数の出力値も返す。
    """

    cached_sums = []  # 記録した全ノードの線形和（Σ）の値
    cached_outs = []  # 記録した全ノードの活性化関数の出力値

    # まずは、入力層を順伝播する
    cached_outs.append(x)  # 何も処理せずに出力値を記録
    next_x = x  # 現在の層の出力（x）＝次の層への入力（next_x）

    # 次に、隠れ層や出力層を順伝播する
    SKIP_INPUT_LAYER = 1
    for layer_i, layer in enumerate(layers):  # 各層を処理
        if layer_i == 0:
            continue  # 入力層は上で処理済み

        # 各層のノードごとに処理を行う
        sums = []
        outs = []
        for node_i in range(layer):  # 層の中の各ノードを処理

            # ノードごとの重みとバイアスを取得
            w = weights[layer_i - SKIP_INPUT_LAYER][node_i]
            b = biases[layer_i - SKIP_INPUT_LAYER][node_i]

            # 【リスト3のコード】ここから↓
            # 1つのノードの処理（1）： 重み付き線形和
            node_sum = summation(next_x, w, b)

            # 1つのノードの処理（2）： 活性化関数
            if layer_i < len(layers)-1:  # -1は出力層以外の意味
                # 隠れ層（シグモイド関数）
                node_out = sigmoid(node_sum)
            else:
                # 出力層（恒等関数）
                node_out = identity(node_sum)
            # 【リスト3のコード】ここまで↑

            # 各ノードの線形和と（活性化関数の）出力をリストにまとめていく
            sums.append(node_sum)
            outs.append(node_out)

        # 各層内の全ノードの線形和と出力を記録
        cached_sums.append(sums)
        cached_outs.append(outs)
        next_x = outs  # 現在の層の出力（outs）＝次の層への入力（next_x）

    if cache_mode:
        return (cached_outs[-1], cached_outs, cached_sums)

    return cached_outs[-1]

y_true = [1.0]  # 正解値
def back_prop(y_true, cached_outs, cached_sums):
    " 逆伝播を行う関数。"
    return None, None

LEARNING_RATE = 0.1 # 学習率（lr）
def update_params(grads_w, grads_b, lr=0.1):
    " パラメーター（重みとバイアス）を更新する関数。"
    return None, None

# ---ここまでは仮の実装。ここからが必要な実装---

# 訓練時の（1）順伝播の実行例
y_pred, cached_outs, cached_sums = forward_prop(*model, x, cache_mode=True)
grads_w, grads_b = back_prop(y_true, cached_outs, cached_sums)  # （2）
weights, biases = update_params(grads_w, grads_b, LEARNING_RATE)  # （3）

print(f'予測値：{y_pred}')  # 予測値： None
print(f'正解値：{y_true}')  # 正解値： [1.0]
