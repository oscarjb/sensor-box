# Generated by Django 2.2.7 on 2020-11-23 16:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0036_auto_20201123_1240'),
    ]

    operations = [
        migrations.AlterField(
            model_name='test',
            name='edited_image',
            field=models.ImageField(upload_to='tests/'),
        ),
    ]
