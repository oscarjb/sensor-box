from django.utils import timezone
from django.db import models
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from jsonfield import JSONField

class MediaFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        if max_length and len(name) > max_length:
            raise (Exception("el nombre del fichero es más grande de lo permitido"))
        return name

    def _save(self, name, content):
        if self.exists(name):
            # si ya existe no se guarda
            return name
        # Si el fichero es nuevo se guarda
        return super(MediaFileSystemStorage, self)._save(name, content)

class Doctor(models.Model):
    user = models.OneToOneField(User, auto_created=True, on_delete=models.CASCADE)
    avatar = models.ImageField(upload_to='', blank=True, default="user.png",
                               storage=MediaFileSystemStorage())
    pin = models.CharField(null=True, max_length=6, blank=True, unique=True)

    def save(self, *args, **kwargs):
        super(Doctor, self).save(*args, **kwargs)
    def __str__(self):
        return self.user.username

class Patient(models.Model):
    name = models.CharField(max_length=20, default="Def name")
    text = models.TextField(default="Def text", blank=True)
    opciones = (("Male", "Male"), ("Female", "Female"), ("Other", "Other"))
    birth_date = models.DateField(blank=True, default="1980-1-1")
    gender = models.CharField(max_length=8, choices=opciones)
    doctor = models.ForeignKey(Doctor, blank=True, on_delete=models.CASCADE)
    photo = models.ImageField(upload_to='', blank=True, default="user.png",
                                    storage=MediaFileSystemStorage())
    def __str__(self):
        return self.name

class Test(models.Model):
    date = models.DateTimeField(default=timezone.now)
    gas_sensors = JSONField(null=True, blank=True)
    distance = models.FloatField(blank=True, default=0.0)
    temperature = JSONField(null=True, blank=True)
    camera_kurokesu = models.ImageField(upload_to='tests/', blank=True)    
    camera_rasp = models.ImageField(upload_to='tests/', blank=True)
    thermal_camera = models.ImageField(upload_to='tests/', blank=True)
    patient = models.ForeignKey(Patient, null=True, blank=True, on_delete=models.CASCADE)
    user = models.ForeignKey(Doctor, null=True, blank=True, on_delete=models.CASCADE)
    opciones = (("Positive", "Positive"), ("Negative", "Negative"))
    outcome = models.CharField(max_length=10, choices=opciones, default="Negative")
    NumberUlcers = models.CharField(max_length=10, choices=opciones, default="0")
    TissueTypes = models.CharField(max_length=50, choices=opciones, default="None")
    Ruler = models.CharField(max_length=10, choices=opciones, default="No")
    ImageRight = models.CharField(max_length=20, choices=opciones, default="Image")
    ImageLeft = models.CharField(max_length=20, choices=opciones, default="Image")
    Imagetherm = models.CharField(max_length=20, choices=opciones, default="Image")
    Perimeter =  models.CharField(max_length=20, choices=opciones, default="0 mm")
    Area =  models.CharField(max_length=20, choices=opciones, default="0 mm2")
    Granulation =  models.CharField(max_length=20, choices=opciones, default="0 %")
    Slough =  models.CharField(max_length=20, choices=opciones, default="0 %")
    Necrosis =  models.CharField(max_length=20, choices=opciones, default="0 %")
    Segmented_leftImage = models.ImageField(blank=True)
    Segmented_leftImage_g = models.ImageField(blank=True)
    Segmented_leftImage_s = models.ImageField(blank=True)
    Segmented_leftImage_n = models.ImageField(blank=True)
    ulcer_edited_image = models.ImageField(upload_to='edited/', blank=True)
    tissueTypes_edited_image = models.ImageField(upload_to='edited/', blank=True)
    giveDistance_edited_image = models.ImageField(upload_to='edited/', blank=True)
    distance_ulcer = models.CharField(max_length=10, choices=opciones, default="0")
    Pixels_in_UPP = models.CharField(max_length=20, choices=opciones, default="0")
    Pixels_in_g = models.CharField(max_length=20, choices=opciones, default="0")
    Pixels_in_s = models.CharField(max_length=20, choices=opciones, default="0")
    Pixels_in_n = models.CharField(max_length=20, choices=opciones, default="0")

    def __str__(self):
        return str(self.date)
