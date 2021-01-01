from django.db import models

# Create your models here.
class Hotel(models.Model):
    print("In Model") 
    name = models.CharField(max_length=50) 
    hotel_Main_Img = models.FileField(upload_to='images/')