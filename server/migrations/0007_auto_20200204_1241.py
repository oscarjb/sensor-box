# Generated by Django 3.0.2 on 2020-02-04 11:41

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0006_auto_20200130_1257'),
    ]

    operations = [
        migrations.AddField(
            model_name='doctor',
            name='pin',
            field=models.CharField(blank=True, max_length=6, null=True, unique=True),
        ),
        migrations.AlterField(
            model_name='test',
            name='date',
            field=models.DateField(default=datetime.datetime(2020, 2, 4, 12, 41, 34, 61886)),
        ),
    ]
