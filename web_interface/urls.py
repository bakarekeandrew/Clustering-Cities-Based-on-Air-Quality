from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('results/', views.clustering_results_view, name='clustering_results'),
]