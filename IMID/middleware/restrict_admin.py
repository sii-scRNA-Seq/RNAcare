from django.http import HttpResponseForbidden

ALLOWED_ADMIN_IPS = ['127.0.0.1','130.209.125.25']  # Replace with allowed IPs

class RestrictAdminMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Check if the request is for the admin panel
        if request.path.startswith('/admin/') and request.META.get('REMOTE_ADDR') not in ALLOWED_ADMIN_IPS:
            return HttpResponseForbidden("You are not allowed to access this page.")
        return self.get_response(request)

