


# 仮の訓練データ（1件分）を準備


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