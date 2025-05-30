# Generated by Django 5.1.6 on 2025-04-27 18:45

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0011_prescription_refill_prescription_status_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='HealthTipCategory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('icon', models.CharField(default='lightbulb', max_length=50)),
                ('description', models.TextField(blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='WellnessArticle',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('content', models.TextField()),
                ('summary', models.TextField()),
                ('read_time', models.PositiveIntegerField(help_text='Estimated reading time in minutes')),
                ('published_date', models.DateField()),
                ('author', models.CharField(blank=True, max_length=100)),
                ('image', models.ImageField(blank=True, null=True, upload_to='articles/')),
            ],
        ),
        migrations.CreateModel(
            name='HealthTip',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('slug', models.SlugField(blank=True, max_length=200, unique=True)),
                ('content', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('is_featured', models.BooleanField(default=False)),
                ('source', models.CharField(blank=True, max_length=200)),
                ('category', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbot.healthtipcategory')),
            ],
        ),
        migrations.CreateModel(
            name='SavedTip',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('saved_at', models.DateTimeField(auto_now_add=True)),
                ('tip', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbot.healthtip')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'unique_together': {('user', 'tip')},
            },
        ),
    ]
