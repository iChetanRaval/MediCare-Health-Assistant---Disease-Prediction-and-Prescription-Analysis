
# Register your models here.

from django.contrib import admin
from .models import CustomUser

admin.site.register(CustomUser)

# admin.py
from django.contrib import admin
from .models import HealthTipCategory, HealthTip, SavedTip, WellnessArticle

from django.contrib import admin
from django.utils.text import slugify
from .models import HealthTipCategory, HealthTip, SavedTip, WellnessArticle

from django.contrib import admin
from .models import HealthTip, HealthTipCategory, SavedTip, WellnessArticle
from django.utils.text import slugify

class HealthTipAdmin(admin.ModelAdmin):
    list_display = ('title', 'category', 'is_featured', 'created_at')
    list_filter = ('category', 'is_featured')
    search_fields = ('title', 'content')
    prepopulated_fields = {'slug': ('title',)}  # Now valid since we added the slug field

    def save_model(self, request, obj, form, change):
        if not obj.slug:
            obj.slug = slugify(obj.title)
        super().save_model(request, obj, form, change)

class WellnessArticleAdmin(admin.ModelAdmin):
    list_display = ('title', 'published_date', 'read_time')
    list_filter = ('published_date',)
    search_fields = ('title', 'content')

# Register all models with their respective admin classes
admin.site.register(HealthTip, HealthTipAdmin)
admin.site.register(HealthTipCategory)
admin.site.register(SavedTip)
admin.site.register(WellnessArticle, WellnessArticleAdmin)  # Only register once with admin class