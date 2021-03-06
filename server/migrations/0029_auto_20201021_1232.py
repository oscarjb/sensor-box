# Generated by Django 2.2.7 on 2020-10-21 10:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0028_auto_20201021_1231'),
    ]

    operations = [
        migrations.AddField(
            model_name='test',
            name='Area',
            field=models.CharField(choices=[('Positive', 'Positive'), ('Negative', 'Negative')], default='0 mm2', max_length=20),
        ),
        migrations.AddField(
            model_name='test',
            name='Granulation',
            field=models.CharField(choices=[('Positive', 'Positive'), ('Negative', 'Negative')], default='0 %', max_length=20),
        ),
        migrations.AddField(
            model_name='test',
            name='ImageLeft',
            field=models.CharField(choices=[('Positive', 'Positive'), ('Negative', 'Negative')], default='Image', max_length=20),
        ),
        migrations.AddField(
            model_name='test',
            name='ImageRight',
            field=models.CharField(choices=[('Positive', 'Positive'), ('Negative', 'Negative')], default='Image', max_length=20),
        ),
        migrations.AddField(
            model_name='test',
            name='Imagetherm',
            field=models.CharField(choices=[('Positive', 'Positive'), ('Negative', 'Negative')], default='Image', max_length=20),
        ),
        migrations.AddField(
            model_name='test',
            name='Necrosis',
            field=models.CharField(choices=[('Positive', 'Positive'), ('Negative', 'Negative')], default='0 %', max_length=20),
        ),
        migrations.AddField(
            model_name='test',
            name='Perimeter',
            field=models.CharField(choices=[('Positive', 'Positive'), ('Negative', 'Negative')], default='0 mm', max_length=20),
        ),
        migrations.AddField(
            model_name='test',
            name='Slough',
            field=models.CharField(choices=[('Positive', 'Positive'), ('Negative', 'Negative')], default='0 %', max_length=20),
        ),
    ]
