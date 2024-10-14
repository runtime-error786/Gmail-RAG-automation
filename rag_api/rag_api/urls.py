from django.urls import path
from core.views import PDFUploadView, query_view

urlpatterns = [
    path('api/upload/', PDFUploadView.as_view(), name='pdf-upload'),
    path('api/query/', query_view, name='query'),
]
