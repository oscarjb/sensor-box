# Generated by Django 3.0.2 on 2020-01-30 11:57

import datetime
from django.db import migrations, models
import server.models


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0005_auto_20200130_1256'),
    ]

    operations = [
        migrations.AlterField(
            model_name='doctor',
            name='avatar',
            field=models.ImageField(blank=True, default='user.png', storage=server.models.MediaFileSystemStorage(), upload_to=''),
        ),
        migrations.AlterField(
            model_name='test',
            name='date',
            field=models.DateField(default=datetime.datetime(2020, 1, 30, 12, 57, 14, 969585)),
        ),
    ]
