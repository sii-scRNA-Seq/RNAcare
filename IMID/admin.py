from django.contrib import admin
from .models import (
    MetaFileColumn,
    CustomUser,
    UploadedFile,
    ProcessFile,
    SharedFile,
    SharedFileInstance,
)


# Register your models here.


@admin.register(MetaFileColumn)
class FileColumnAdmin(admin.ModelAdmin):
    list_display = ("user", "cID", "colName", "label", "numeric")


@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = (
        "username",
        "email",
        "first_name",
        "last_name",
        "is_staff",
        "is_active",
        "date_joined",
        "uuid",
    )


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ("user", "cID", "type1", "filename", "label")


@admin.register(ProcessFile)
class FileColumnAdmin(admin.ModelAdmin):
    list_display = ("user", "cID", "pickle_file")


@admin.register(SharedFile)
class SharedFileAdmin(admin.ModelAdmin):
    list_display = ("user", "cohort", "type1", "filename", "label")


@admin.register(SharedFileInstance)
class SharedFileInstanceAdmin(admin.ModelAdmin):
    list_display = ("user", "filename", "label")
