from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('convert2text/', views.convert2text, name='convert2text'),
    path('summarization/', views.summarization, name='summarization'),
]