from django.shortcuts import redirect

ALLOWED_ADMIN_IPS = ['127.0.0.1','130.209.125.25']  # Replace with allowed IPs

class RestrictAdminMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Check if the request is for the admin panel
        ip = request.META.get('HTTP_X_FORWARDED_FOR')
        if ip:
            ip = ip.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR')
        if request.path.startswith('/admin/') and ip not in ALLOWED_ADMIN_IPS:
            return redirect('/accounts/login/')
        return self.get_response(request)

