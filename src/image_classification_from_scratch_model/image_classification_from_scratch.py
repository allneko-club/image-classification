"""
# Introduction

https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_from_scratch.py

この例では、事前にトレーニングされた重みや事前に作成された Keras アプリケーションモデルを利用せずに、
ディスク上の jpeg画像ファイルを使って、画像分類を行う方法を示します。
Kaggle Cats vs Dogs バイナリ分類データセットのワークフローを紹介ます。

image_dataset_from_directory ユーティリティを使用してデータセットを生成し、
Keras 画像前処理レイヤーを使用して画像の標準化とデータ拡張を行います。
"""
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
## Load the data: the Cats vs Dogs dataset
### 786M のzipデータをダウンロード

curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
unzip -q kagglecatsanddogs_5340.zip

CatとDogフォルダーを含むPetImagesフォルダーが現れる. 各サブフォルダーには、各カテゴリの画像ファイルが含まれている。
"""

"""
### 破損した画像を除外する
多くの実世界の画像データを扱う場合、破損した画像がよく発生します。
ヘッダーに文字列「JFIF」が含まれていない不適切にエンコードされた画像を除外
"""

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

"""
## Generate a `Dataset`
"""

image_size = (180, 180)
batch_size = 128

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

"""
## データを視覚化する
トレーニング データセットの最初の 9 つの画像を次に示します。
ご覧のとおり、ラベル 1 は「犬」、ラベル 0 は「猫」です
"""

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

"""
## 画像データ拡張の使用
大規模な画像データセットがない場合は、ランダムでありながら現実的な変換 (ランダムな水平方向の反転や小さなランダムな回転など) 
をトレーニング画像に適用することで、人為的にサンプルの多様性を導入することをお勧めします。
これにより、モデルをトレーニング データのさまざまな側面にさらしながら、オーバーフィッティングを遅くすることができます。
"""

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

"""
data_augmentation データセットの最初の画像に繰り返し適用して、
拡張されたサンプルがどのように見えるかを視覚化しましょう。
"""

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

"""
## データの標準化
データセットによって連続したfloat32バッチとして生成されるため、画像は既に標準サイズ (180x180) になっています。
ただし、それらの RGB チャネル値は[0, 255]範囲内にあります。
これは、ニューラル ネットワークには理想的ではありません。
一般に、入力値を小さくするように努める必要があります。
ここでは、モデルの開始時にRescalingレイヤー[0, 1]を使用して、値を標準化します。
"""

"""
## データを前処理するための 2 つのオプション
`data_augmentation`プリプロセッサを使用する方法は 2 つあります
Option 1: 次のように、モデルの一部にします:
```python
inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
...  # Rest of the model
```
このオプションを使用すると、データ拡張がdevice で発生し、残りのモデル実行と同期します。
つまり、GPU アクセラレーションの恩恵を受けます。

データ拡張はテスト時に非アクティブであるため、入力サンプルは、evaluate() や predict() を呼び出すときではなく、
fit() 中にのみ拡張されることに注意してください。
GPU でトレーニングしている場合、これは良いオプションかもしれません。

Option 2: 次のように、拡張画像のバッチを生成するデータセットを取得するために、データセットに適用します
```python
augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))
```
このオプションを使用すると、データ拡張はCPU で非同期に発生し、モデルに入る前にバッファリングされます。
CPU でトレーニングしている場合は、データ拡張が非同期でノンブロッキングになるため、これがより良いオプションです。
この場合、2 番目のオプションを使用します。どちらを選択するかわからない場合は、
この 2 番目のオプション (非同期前処理) が常に確実な選択です。
"""

"""
## パフォーマンスのためにデータセットを構成する
トレーニング データセットにデータ拡張を適用し、I/O がブロックされることなくディスクからデータを取得できるように、
バッファ付きプリフェッチを使用するようにする。
"""

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

"""
## Build a model
モデルを構築する
Xception ネットワークの小さなバージョンを構築します。アーキテクチャの最適化は特に試みていません。
最適なモデル構成を体系的に検索したい場合は、 KerasTuner の使用を検討してください。
[KerasTuner](https://github.com/keras-team/keras-tuner).

留意点:
- モデルはdata_augmentationプリプロセッサで開始し、その後に Rescalingレイヤーが続きます。
- Dropout最終的な分類レイヤーの前にレイヤーを含めます。
"""


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)

"""
## モデルをトレーニングする
"""

epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

"""
完全なデータセットで 25 エポックのトレーニングを行った後、90% を超える検証精度が得られます.
(実際には、検証パフォーマンスが低下し始める前に 50 エポック以上のトレーニングを行うことができます)
"""

"""
## 新しいデータで推論を実行する
データの拡張とドロップアウトは、推論時には非アクティブであることに注意してください。
"""

img = keras.preprocessing.image.load_img(
    "PetImages/Cat/6779.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
