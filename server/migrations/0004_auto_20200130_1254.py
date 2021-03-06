# Generated by Django 3.0.2 on 2020-01-30 11:54

import datetime
from django.db import migrations, models
import server.models


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0003_test'),
    ]

    operations = [
        migrations.AlterField(
            model_name='patient',
            name='photo',
            field=models.ImageField(blank=True, default='server/static/images/user.png', storage=server.models.MediaFileSystemStorage(), upload_to='server/static/images'),
        ),
        migrations.AlterField(
            model_name='test',
            name='date',
            field=models.DateField(default=datetime.datetime(2020, 1, 30, 12, 54, 7, 706454)),
        ),
    ]
