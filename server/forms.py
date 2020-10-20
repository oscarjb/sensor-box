from django import forms
from .models import *
from django.contrib.auth.forms import UserCreationForm
from betterforms.multiform import MultiModelForm


class UserForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'username', 'email', 'password1', 'password2',)


class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ('name', 'gender', 'photo')

    def clean_photo(self):
        photo = self.cleaned_data['photo']
        try:
            main, sub = photo.content_type.split('/')
            if not (main == 'image' and sub in ['jpeg', 'pjpeg', 'gif', 'png']):
                raise forms.ValidationError(u'Please use a JPEG, '
                    'GIF or PNG image.')
        except AttributeError:
            pass
        return photo

class DoctorForm(forms.ModelForm):
    class Meta:
        model = Doctor
        fields = ('avatar', 'pin', )

    def clean_avatar(self):
        avatar = self.cleaned_data['avatar']
        try:
            main, sub = avatar.content_type.split('/')
            if not (main == 'image' and sub in ['jpeg', 'pjpeg', 'gif', 'png']):
                raise forms.ValidationError(u'Please use a JPEG, '
                    'GIF or PNG image.')
        except AttributeError:
            pass
        return avatar


class User_Doctor_Form(MultiModelForm):
    form_classes = {
        'user':UserForm,
        'doctor': DoctorForm,
    }