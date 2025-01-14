from django.urls import path, include
from .views import signup_view
from .views import CustomLoginView

urlpatterns=[
    path('signup/',signup_view,name='signup'),
    path(
        "login/",
        CustomLoginView.as_view(redirect_authenticated_user=True),
        name="login",
    ),
    path("captcha/", include("captcha.urls")),
]
