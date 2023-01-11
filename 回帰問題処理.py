import random
import Dnn_Forward_Prop
import Dnn_Back_Prop
import Dnn_Node_process
import matplotlib.pyplot as plt


def accumulate(list1, list2):
    "2つのリストの値を足し算する関数。"
    new_list = []
    for item1, item2 in zip(list1, list2):
        if isinstance(item1, list):
            child_list = accumulate(item1, item2)
            new_list.append(child_list)
        else:
            new_list.append(item1 + item2)
    return new_list

def mean_element(list1, data_count):
    "1つのリストの値をデータ数で平均する関数。"
    new_list = []
    for item1 in list1:
        if isinstance(item1, list):
            child_list = mean_element(item1, data_count)
            new_list.append(child_list)
        else:
            new_list.append(item1 / data_count)
    return new_list


def optimize(model, x, y, data_i, last_i, batch_i, batch_size, acm_g, lr=0.1):
    "train()親関数から呼ばれる、最適化のための子関数。"

    layers = model[0]  # レイヤー構造
    each_x = x[data_i]  # 1件分の訓練データ
    y_true = y[data_i]  # 1件分の正解値

    # ステップ（1）順伝播
    y_pred, outs, sums = Dnn_Forward_Prop.forward_prop(*model, each_x, cache_mode=True)

    # ステップ（2）逆伝播
    gw, gb = Dnn_Back_Prop.back_prop(*model, y_true, outs, sums)

    # 各勾配を蓄積（accumulate）していく
    if batch_i == 0:
        acm_gw = gw
        acm_gb = gb
    else:
        acm_gw = accumulate(acm_g[0], gw)
        acm_gb = accumulate(acm_g[1], gb)
    batch_i += 1  # バッチ番号をカウントアップ＝現在のバッチ数

    # 訓練状況を評価するために、損失値を取得
    loss = 0.0
    for output, target in zip(y_pred, y_true):
        loss += Dnn_Node_process.sseloss(output, target)

    # バッチサイズごとで後続の処理に進む
    if batch_i % BATCH_SIZE != 0 and data_i != last_i:
        return model, loss, batch_i, (acm_gw, acm_gb)  # バッチ内のデータごと

    layers = model[0]  # レイヤー構造
    out_count = layers[-1]  # 出力層のノード数

    # 平均二乗誤差なら平均する（損失関数によって異なる）
    grads_w = mean_element(acm_gw, batch_i * out_count)  # 「バッチサイズ ×
    grads_b = mean_element(acm_gb, batch_i * out_count)  # 　出力ノード数」で平均
    batch_i = 0  # バッチ番号を初期化して次のイテレーションに備える

    # ステップ（3）パラメーター（重みとバイアス）の更新
    weights, biases = update_params(*model, grads_w, grads_b, lr)

    # モデルをアップデート（＝最適化）
    model = (layers, weights, biases)

    return model, loss, batch_i, (acm_gw, acm_gb)  # イテレーションごと

def update_params(layers, weights, biases, grads_w, grads_b, lr=0.1):
    """
    パラメーター（重みとバイアス）を更新する関数。
    - 引数：
    (layers, weights, biases): モデルを指定する。
    grads_w: 重みの勾配。
    grads_b: バイアスの勾配。
    lr: 学習率（learning rate）。最適化を進める量を調整する。
    - 戻り値:
    新しい重みとバイアスを返す。
    """

    # ネットワーク全体で勾配を保持するためのリスト
    new_weights = [] # 重み
    new_biases = [] # バイアス

    SKIP_INPUT_LAYER = 1
    for layer_i, layer in enumerate(layers):  # 各層を処理
        if layer_i == 0:
            continue  # 入力層はスキップ

        # 層ごとで勾配を保持するためのリスト
        layer_w = []
        layer_b = []

        for node_i in range(layer):  # 層の中の各ノードを処理
            b = biases[layer_i - SKIP_INPUT_LAYER][node_i]
            grad_b = grads_b[layer_i - SKIP_INPUT_LAYER][node_i]
            b = b - lr * grad_b  # バイアスパラメーターの更新
            layer_b.append(b)

            node_weights = weights[layer_i - SKIP_INPUT_LAYER][node_i]
            node_w = []
            for each_w_i, w in enumerate(node_weights):
                grad_w = grads_w[layer_i - SKIP_INPUT_LAYER][node_i][each_w_i]
                w = w - lr * grad_w  # 重みパラメーターの更新
                node_w.append(w)
            layer_w.append(node_w)

        new_weights.append(layer_w)
        new_biases.append(layer_b)
    
    return (new_weights, new_biases)


def train(model, x, y, batch_size=32, epochs=10, lr=0.1, verbose=10):
    """
    モデルの訓練を行う関数（親関数）。
    - 引数：
    model: モデルをタプル「(layers, weights, biases)」で指定する。
    x: 訓練データ（各データが行、各特徴量が列の、2次元リスト値）。
    y: 訓練ラベル（各データが行、各正解値が列の、2次元リスト値）。
    batch_size: バッチサイズ。何件のデータをまとめて処理するか。
    epochs: エポック数。全データ分で何回、訓練するか。
    lr: 学習率（learning rate）。最適化を進める量を調整する。
    verbose: 訓練状況を何エポックおきに出力するか。
    - 戻り値:
    損失値の履歴を返す。これを使って損失値の推移グラフが描ける。
    """
    loss_history = []  # 損失値の履歴

    data_size = len(y)  # 訓練データ数
    data_indexes = range(data_size)  # 訓練データのインデックス

    # 各エポックを処理
    for epoch_i in range(1, epochs + 1):  # 経過表示用に1スタート

        acm_loss = 0  # 損失値を蓄積（accumulate）していく

        # 訓練データのインデックスをシャッフル（ランダムサンプリング）
        random_indexes = random.sample(data_indexes, data_size)
        last_i = random_indexes[-1]  # 最後の訓練データのインデックス

        # 親関数で管理すべき変数
        acm_g = (None, None)  # 重み／バイアスの勾配を蓄積していくため
        batch_i = 0  # バッチ番号をインクリメントしていくため

        # 訓練データを1件1件処理していく
        for data_i in random_indexes:

            # 親子に分割したうちの子関数を呼び出す
            model, loss, batch_i, acm_g = optimize(
                model, x, y, data_i, last_i, batch_i, batch_size, acm_g, lr)

            acm_loss += loss  # 損失値を蓄積

        # エポックごとに損失値を計算。今回の実装では「平均」する
        layers = model[0]  # レイヤー構造
        out_count = layers[-1]  # 出力層のノード数
        # 「訓練データ数（イテレーション数×バッチサイズ）×出力ノード数」で平均
        epoch_loss = acm_loss / (data_size * out_count)

        # 訓練状況を出力
        if verbose != 0 and \
            (epoch_i % verbose == 0 or epoch_i == 1 or epoch_i == EPOCHS):
            print(f'[Epoch {epoch_i}/{EPOCHS}] train_loss: {epoch_loss}')

        loss_history.append(epoch_loss)  # 損失値の履歴として保存

    return model, loss_history



# 訓練データを取得
import plygdata as pg
PROBLEM_DATA_TYPE = pg.DatasetType.RegressPlane
TRAINING_DATA_RATIO = 0.5
DATA_NOISE = 0.0
data_list = pg.generate_data(PROBLEM_DATA_TYPE, DATA_NOISE)
X_train, y_train, _, _ = pg.split_data(data_list, training_size=TRAINING_DATA_RATIO)


# モデルを定義
layers = [2, 3, 1]
weights = [
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0, 0.0]]
]
biases = [
    [0.0, 0.0, 0.0],  # hidden1
    [0.0]  # output
]
model = (layers, weights, biases)

# 訓練用のハイパーパラメーター設定
BATCH_SIZE = 4   # バッチサイズ
EPOCHS = 100     # エポック数
LEARNING_RATE = 0.02  # 学習係数

# 訓練処理の実行
model, loss_history = train(model, X_train, y_train, BATCH_SIZE, EPOCHS, LEARNING_RATE)

# 学習結果（損失）のグラフを描画
epochs = len(loss_history)
plt.plot(range(1, epochs + 1), loss_history, marker='.', label='loss (Training data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()