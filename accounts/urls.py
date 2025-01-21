from django.urls import path, include
from .views import signup_view
from .views import CustomLoginView
from django.contrib.auth import views as auth_views

urlpatterns=[
    path('signup/',signup_view,name='signup'),
    path(
        "login/",
        CustomLoginView.as_view(redirect_authenticated_user=True),
        name="login",
    ),
    path("captcha/", include("captcha.urls")),
    path('password_reset/', auth_views.PasswordResetView.as_view(), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),
]
