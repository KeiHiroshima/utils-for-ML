import os
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


# ROC 曲線を作図する関数
def plot_roc_curve(
    false_pos_rate,  # 各閾値ごとの偽陽性率
    true_pos_rate,  # 各閾値ごとの真陽性率
    best_threthold_idx=None,
    log_path="./logs",
):  # 最適な閾値のインデックス
    # ROC 曲線の作図
    plt.plot(false_pos_rate, true_pos_rate, color="red", label="ROC")
    # 斜め 45 度線の作図
    plt.plot([0, 1], [0, 1], color="green", linestyle="--")
    # 最適な閾値が与えられたときの処理
    if best_threthold_idx is not None:
        # 最適な閾値に対応する偽陽性率の縦線を作図
        plt.axvline(
            x=false_pos_rate[best_threthold_idx],
            ymin=0,
            ymax=1,
            color="blue",
            linestyle=":",
            label="最適な閾値",
        )
        # 最適な閾値に対応する真陽性率の横線を作図
        plt.axhline(
            y=true_pos_rate[best_threthold_idx],
            xmin=0,
            xmax=1,
            color="blue",
            linestyle=":",
        )
        # 最適な閾値に対応する偽陽性率・真陽性率の点を作図
        plt.scatter(
            x=false_pos_rate[best_threthold_idx],
            y=true_pos_rate[best_threthold_idx],
            color="blue",
        )
    # ラベルの設定
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # 図の表示
    # plt.legend()
    plt.savefig(os.path.join(log_path, "roc.png"))
    plt.show()


# テストデータに対する評価を行う関数
def evaluate(y_pred_proba, y_pred, y_test, dataset_name, log_path):
    # 混同行列の集計
    cm = confusion_matrix(
        y_test,  # テストデータの目的変数
        y_pred,  # テストデータに対する予測結果
        labels=[True, False],
    )  # 行・列ともに True → False の順番で集計

    # 混同行列をPandasのデータフレームに整形
    cm_df = pd.DataFrame(
        cm, index=["実測_悪性", "実測_良性"], columns=["予測_悪性", "予測_良性"]
    )

    # 混同行列の表示
    print("==== 混同行列 ====")
    print(cm_df)

    print("==== 評価指標 ====")
    # 正答率の計算と表示
    accuracy = accuracy_score(y_test, y_pred)
    print(f"テストデータの正答率: {accuracy:.2%}")

    # 適合率・再現率・F1スコアの計算と表示
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test,  # テストデータの目的変数
        y_pred,  # テストデータに対する予測結果
        labels=[True, False],
    )  # True → False の順番で計算
    print(f"テストデータの適合率: {precision[0]:.2%}")
    print(f"テストデータの再現率: {recall[0]:.2%}")
    print(f"テストデータのF1スコア: {f1_score[0]:.2%}")

    # AUC の計算と表示
    roc_auc = roc_auc_score(
        y_test,  # テストデータの目的変数
        y_pred_proba,
    )  # テストデータに対する、予測が True となる確率
    print(f"テストデータのAUC: {roc_auc:.2%}")

    # print('==== ROC 曲線 ====')
    # 偽陽性率・真陽性率・閾値の計算
    false_pos_rate, true_pos_rate, thretholds = roc_curve(
        y_test,  # テストデータの目的変数
        y_pred_proba,
    )  # テストデータに対する、予測が True となる確率
    # ROC 曲線の描画
    plot_roc_curve(false_pos_rate, true_pos_rate, log_path=log_path)

    # add accuracy, precision, recall, f1_score, roc_auc values to result.txt
    with open(os.path.join(log_path, "result-" + dataset_name + ".txt"), mode="a") as f:
        f.write(f"{accuracy}, {precision[0]}, {recall[0]}, {f1_score[0]}, {roc_auc}\n")

    return f1_score[0]


def threshold_search(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    func: Callable[[np.ndarray, np.ndarray], float],
    is_higher_better: bool = True,
) -> Dict[str, float]:
    best_threshold = 0.0
    best_score = -np.inf if is_higher_better else np.inf

    for threshold in [i * 0.01 for i in range(100)]:
        score = func(y_true=y_true, y_pred=y_proba > threshold)
        if is_higher_better:
            if score > best_score:
                best_threshold = threshold
                best_score = score
        else:
            if score < best_score:
                best_threshold = threshold
                best_score = score

    search_result = {"threshold": best_threshold, "score": best_score}
    return search_result
