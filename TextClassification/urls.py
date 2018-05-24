from django.urls import path

from . import views

urlpatterns = [
    path('classification', views.TextClassification.as_view(), name='text-classification'),
]
