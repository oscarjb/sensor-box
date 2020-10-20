from django.contrib import admin
from .models import Doctor, Patient, Test
from django.db import models
# Register your models here.

class DoctorAdmin(admin.ModelAdmin):
    list_display = ('user', )
    list_filter = ['user']
    search_fields = ['user']

class PatientAdmin(admin.ModelAdmin):
    list_display = ('name', 'doctor')
    list_filter = ['name']
    search_fields = ['name']

class TestAdmin(admin.ModelAdmin):
    list_display = ('date', 'patient')
    list_filter = ['date']
    search_fields = ['date']

admin.site.register(Doctor, DoctorAdmin)
admin.site.register(Patient, PatientAdmin)
admin.site.register(Test, TestAdmin)