# forms.py 
from django import forms 
from .models import *
  
class HotelForm(forms.ModelForm):
  
    class Meta: 
        print("In Form")
        model = Hotel 
        fields = ['name', 'hotel_Main_Img']