"""Ulcers_Server URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from server.views import *
from Ulcers_Server import settings
from django.conf.urls.static import static
from django.conf.urls import url

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', profile, name="index"),
    path('profile/', profile, name="profileView"),
    path('users/<int:patient_id>', search_users, name="search_users"),
    path('edit/<int:patient_id>', edit_patient, name="edit_patient"),
    path('users/<int:patient_id>/<int:test_id>', search_tests, name="search_tests"),
    path('remove/<int:patient_id>', remove_patient, name="remove_patient"),
    path('remove/<int:patient_id>/<int:test_id>', remove_test, name="remove_test"),
    path('edit/', edit_user, name="edit_user"),
    path('logout/', logout, name="logout"),
    path('create/', create_patient, name="create_patient"),
    path('login/', login_register, name="login_register"),
    path('api/data', recieve_data, name="recieve_data"),
    path('api/login', login_api, name="login_api"),
    url(r'^ajax/analyze_data/$', analyze_data, name='analyze_data'),
    url(r'^ajax/save_image/$', save_image, name='save_image'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
