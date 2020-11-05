# Generated by Django 2.2.7 on 2020-02-28 10:04

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0009_auto_20200213_1159'),
    ]

    operations = [
        migrations.AddField(
            model_name='test',
            name='outcome',
            field=models.CharField(choices=[('Positive', 'Positive'), ('Negative', 'Negative')], default='Negative', max_length=10),
        ),
        migrations.AlterField(
            model_name='test',
            name='date',
            field=models.DateField(default=datetime.datetime(2020, 2, 28, 11, 4, 36, 642850)),
        ),
    ]
