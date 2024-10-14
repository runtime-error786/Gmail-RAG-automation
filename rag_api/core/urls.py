from django.urls import path
from .views import PDFUploadView, QueryView

urlpatterns = [
    path('upload/', PDFUploadView.as_view(), name='upload_pdf'),
    path('query/', QueryView.as_view(), name='query_rag'),
]
