from django.core.exceptions import ValidationError


def validate_image_size(image):
    limit = 1 * 1024 * 1024  # 1MB
    if limit < image.size:
        raise ValidationError("画像サイズが大きすぎます。1MB以下の画像ファイルを選択して下さい。")
