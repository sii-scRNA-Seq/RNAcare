from django.contrib import admin
from .models import MetaFileColumn

# Register your models here.


@admin.register(MetaFileColumn)
class FileColumnAdmin(admin.ModelAdmin):
    list_display = ("user", "cID", "colName", "label")
