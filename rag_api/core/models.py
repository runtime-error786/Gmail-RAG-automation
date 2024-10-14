from django.db import models

class PDFUpload(models.Model):
    pdf = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
class Query(models.Model):
    query_text = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
