import logging
import os
import pathlib
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__file__)


class FoodsModel:

    def __init__(self, model_name):
        self.data_dir = pathlib.Path(f"{BASE_DIR}/{model_name}/resources/")
        self.saved_model_path = f"{BASE_DIR}/{model_name}/output"

    def get_classe_names(self):
        """分類名を全て取得する"""
        train_ds = tf.keras.utils.image_dataset_from_directory(self.data_dir)
        return train_ds.class_names

    def predict(self, image_path):
        img_height = 180
        img_width = 180

        model = tf.keras.models.load_model(self.saved_model_path)

        img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        return model.predict(img_array)


def _get_ranking(score, class_names):
    """
    scoreとclass_namesから、スコアが高いランキングデータを作成する。
    :param score: 1次元の ndarray インスタンス
    :param class_names: 分類名のリスト
    :return:[(1位の分類名, 1位のスコア), (2位の分類名, 2位のスコア), ...] のリスト
    """
    scores = score.tolist()  # numpyはforループで使えないため、pythonのlist形式に変換
    ranks = score.argsort()[::-1].argsort().tolist()  # scoreが高い順に順位付けしたリストを作成

    class_name_score_list = [None] * len(ranks)
    for i, rank in enumerate(ranks):
        class_name_score_list[rank] = (class_names[i], scores[i])
    return class_name_score_list


def predict(image, model_name="foods_model"):
    """
    imageの分類結果を返す
    :param image:
    :param model_name: モデル名 (ml/ のフォルダ名)
    :return:
    """
    foods_model = FoodsModel(model_name)
    class_names = foods_model.get_classe_names()
    logger.info("class names: %s", class_names)

    predictions = foods_model.predict(image)
    score = tf.nn.softmax(predictions[0])
    class_name_score_list = _get_ranking(score.numpy(), class_names)
    return class_name_score_list
