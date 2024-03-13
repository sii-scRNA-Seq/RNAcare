from django.contrib import admin
from .models import MetaFileColumn

# Register your models here.


@admin.register(MetaFileColumn)
class FileColumnAdmin(admin.ModelAdmin):
    list_display = ("cID", "colName", "label")


# @admin.register(CustomUser)
# class CustomUserAdmin(admin.ModelAdmin):
#     list_display = ("cID", "pickle_file")


# @admin.register(UploadedFile)
# class UploadedFileAdmin(admin.ModelAdmin):
#     list_display = ("cID", "type", "filename", "label")
