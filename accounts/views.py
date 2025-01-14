from django.shortcuts import render, redirect
from django import forms
from django.contrib.auth.forms import UserCreationForm
from IMID.models import CustomUser
from django.core.exceptions import ValidationError
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.contrib.auth.views import LoginView
from .forms import CustomLoginForm
from captcha.fields import CaptchaField


# Create your views here.
class CustomLoginView(LoginView):
    form_class = CustomLoginForm
    template_name = "registration/login.html"


def validate_username(value):
    if "_" in value:
        raise ValidationError("Usernames cannot contain underscores.")


class SignUpForm(UserCreationForm):
    username = forms.CharField(
        max_length=30,
        validators=[validate_username],
        help_text="Required. not allowed '_'",
    )
    email = forms.EmailField(
        max_length=100, help_text="Required. Enter a valid email address."
    )
    captcha = CaptchaField()

    class Meta:
        model = CustomUser
        fields = ("username", "email", "password1", "password2")


def signup_view(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    else:
        form = SignUpForm()
    return render(request, "registration/signup.html", {"form": form})
