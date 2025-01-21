# IMID/middleware/dynamic_csrf_middleware.py
from django.middleware.csrf import CsrfViewMiddleware
from django.utils.deprecation import MiddlewareMixin

class DynamicCsrfMiddleware(MiddlewareMixin):
    def process_view(self, request, view_func, view_args, view_kwargs):
        if (
            "X-Requested-With" in request.headers
            and request.headers["X-Requested-With"] == "XMLHttpRequest"
        ):
            setattr(request, "_dont_enforce_csrf_checks", True)
        return None
