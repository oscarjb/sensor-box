# Generated by Django 2.2.7 on 2020-04-28 20:39

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0020_auto_20200428_2039'),
    ]

    operations = [
        migrations.AlterField(
            model_name='test',
            name='date',
            field=models.DateTimeField(default=datetime.datetime(2020, 4, 28, 20, 39, 24, 671793)),
        ),
    ]
