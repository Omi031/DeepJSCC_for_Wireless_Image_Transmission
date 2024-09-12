# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 画像を作成またはロードする (例としてランダムなデータを使用)
# [10, 32, 32, 3]の形状を持つランダムな画像データ
# images = np.random.rand(10, 32, 32, 3)


def save_image_grid(images, rows, cols, save_path):
    """
    画像をグリッド形式で並べて保存する
    images: [N, H, W, C]の形状を持つ画像配列
    rows: グリッドの行数
    cols: グリッドの列数
    save_path: 保存先のファイルパス
    """
    fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))

    # グリッドに画像を埋めていく
    for i in range(rows * cols):
        if i < images.shape[0]:
            ax = axs[i // cols, i % cols]
            ax.imshow(images[i])
            ax.axis("off")  # 軸を非表示にする
        else:
            axs[i // cols, i % cols].axis("off")  # 画像がない部分は空白にする

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(save_path)
    # plt.show()


# 行と列の数を指定して画像を並べる
# rows, cols = 2, 5  # 2行5列で10枚の画像を並べる
# save_path = "image_grid.png"
# save_image_grid(images, rows, cols, save_path)
