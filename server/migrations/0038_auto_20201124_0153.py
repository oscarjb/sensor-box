# Generated by Django 2.2.7 on 2020-11-24 00:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0037_auto_20201123_1722'),
    ]

    operations = [
        migrations.AlterField(
            model_name='test',
            name='edited_image',
            field=models.ImageField(upload_to='edited/'),
        ),
    ]
