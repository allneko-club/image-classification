from django import forms

from ai.classification.models import ClassifiedImage


class ClassifyImageForm(forms.ModelForm):

    class Meta:
        model = ClassifiedImage
        fields = ("image",)
        widgets = {
            "image": forms.ClearableFileInput(attrs={"class": "form-control"}),
        }

    # def clean_image(self):
    #     images = self.files.getlist('image')
    #     for image in images:
    #         validate_image_size(image)
