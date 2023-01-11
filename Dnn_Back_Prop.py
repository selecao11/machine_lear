import Dnn_Node_process
def back_prop(layers, weights, biases, y_true, cached_outs, cached_sums):
    """
    逆伝播を行う関数。
    - 引数：
    (layers, weights, biases): モデルを指定する。
    y_true: 正解値（出力層のノードが複数ある場合もあるのでリスト値）。
    cached_outs: 順伝播で記録した活性化関数の出力値。予測値を含む。
    cached_sums: 順伝播で記録した線形和（Σ）値。
    - 戻り値：
    重みの勾配とバイアスの勾配を返す。
    """

    # ネットワーク全体で勾配を保持するためのリスト
    grads_w =[]  # 重みの勾配
    grads_b = []  # バイアスの勾配
    grads_x = []  # 入力の勾配

    layer_count = len(layers)
    layer_max_i = layer_count-1
    SKIP_INPUT_LAYER = 1
    PREV_LAYER = 1
    rng = range(SKIP_INPUT_LAYER, layer_count)  # 入力層以外の層インデックス
    for layer_i in reversed(rng):  # 各層を逆順に処理

        is_output_layer = (layer_i == layer_max_i)
        # 層ごとで勾配を保持するためのリスト
        layer_grads_w = []
        layer_grads_b = []
        layer_grads_x = []

        # （1）逆伝播していく誤差情報
        if is_output_layer:
            # 出力層（損失関数の偏微分係数）
            back_error = []  # 逆伝播していく誤差情報
            y_pred = cached_outs[layer_i]
            for output, target in zip(y_pred, y_true):
                loss_der = Dnn_Node_process.sseloss_der(output, target)  # 誤差情報
                back_error.append(loss_der)# 誤差格納
        else:
            # 隠れ層（次の層への入力の偏微分係数）
            back_error = grads_x[-1]  # 最後に追加された入力の勾配

        node_sums = cached_sums[layer_i - SKIP_INPUT_LAYER]
        for node_i, node_sum in enumerate(node_sums):  # 各ノードを処理

            # （2）活性化関数を偏微分
            if is_output_layer:
                # 出力層（恒等関数の微分）
                active_der = Dnn_Node_process.identity_der(node_sum)
            else:
                # 隠れ層（シグモイド関数の微分）
                active_der = Dnn_Node_process.sigmoid_der(node_sum)

            # （3）線形和を重み／バイアス／入力で偏微分
            w = weights[layer_i - SKIP_INPUT_LAYER][node_i]
            b = biases[layer_i - SKIP_INPUT_LAYER][node_i]
            x = cached_outs[layer_i - PREV_LAYER]  # 前の層の出力＝今の層への入力
            sum_der_w = Dnn_Node_process.sum_der(x, w, b, with_respect_to='w')
            sum_der_b = Dnn_Node_process.sum_der(x, w, b, with_respect_to='b')
            sum_der_x = Dnn_Node_process.sum_der(x, w, b, with_respect_to='x')

            # （4）各重み／バイアス／各入力の勾配を計算
            delta = back_error[node_i] * active_der

            # バイアスは1つだけ
            grad_b = delta * sum_der_b
            layer_grads_b.append(grad_b)

            # 重みと入力は前の層のノードの数だけある
            node_grads_w = []
            for x_i, (each_dw, each_dx) in enumerate(zip(sum_der_w, sum_der_x)):
                # 重みは個別に取得する
                grad_w = delta * each_dw
                node_grads_w.append(grad_w)

                # 入力は各ノードから前のノードに接続する全ての入力を合計する
                # （※重み視点と入力視点ではエッジの並び方が違うので注意）
                grad_x = delta * each_dx
                if node_i == 0:
                    # 最初に、入力の勾配を作成
                    layer_grads_x.append(grad_x)
                else:
                    # その後は、その入力の勾配に合計していく
                    layer_grads_x[x_i] += grad_x
            layer_grads_w.append(node_grads_w)

        # 層ごとの勾配を、ネットワーク全体用のリストに格納
        grads_w.append(layer_grads_w)
        grads_b.append(layer_grads_b)
        grads_x.append(layer_grads_x)

    # 保持しておいた各勾配（※逆順で追加したので反転が必要）を戻り値で返す
    grads_w.reverse()
    grads_b.reverse()
    return (grads_w, grads_b)  # grads_xは最適化で不要なので返していない
