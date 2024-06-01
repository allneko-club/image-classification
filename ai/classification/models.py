from django.db import models
from ai.classification.exceptions import validate_image_size


class ClassifiedImage(models.Model):
    image = models.ImageField(
        "分類画像",
        upload_to='images/%Y/%m/%d/',
        validators=[validate_image_size]
    )
    result = models.CharField(max_length=50, blank=True, default="")
    date_created = models.DateTimeField(auto_now_add=True)
