# Generated by Django 2.2.7 on 2020-10-21 10:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0024_auto_20201021_1147'),
    ]

    operations = [
        migrations.AlterField(
            model_name='test',
            name='ImageLeft',
            field=models.CharField(choices=[('Positive', 'Positive'), ('Negative', 'Negative')], default='Image', max_length=20),
        ),
        migrations.AlterField(
            model_name='test',
            name='ImageRight',
            field=models.CharField(choices=[('Positive', 'Positive'), ('Negative', 'Negative')], default='Image', max_length=20),
        ),
        migrations.AlterField(
            model_name='test',
            name='Imagetherm',
            field=models.CharField(choices=[('Positive', 'Positive'), ('Negative', 'Negative')], default='Image', max_length=20),
        ),
    ]
