from django.contrib import messages
from django.views.generic import FormView
from django.urls import reverse_lazy

from .forms import ClassifyImageForm
from ai.ml.image_classification import predict


class HomeView(FormView):
    form_class = ClassifyImageForm
    template_name = "classification/home.html"
    success_url = reverse_lazy("home")

    def form_valid(self, form):
        obj = form.save()
        class_name_score_list = predict(obj.image.path, "foods_model")
        for rank, class_name_score in enumerate(class_name_score_list):
            print(f"{rank + 1}位 {class_name_score[0]} スコア：{class_name_score[1]}")

        messages.info(self.request, f"結果：{class_name_score_list}")
        return super().form_valid(form)
