from django.http import HttpResponse 
from django.shortcuts import render, redirect 
from .forms import *
from rest_framework.views import APIView 
from .apps import FastbertConfig

# Create your views here. 
class call_model(APIView):

    def post(self, request):                 

        print("In View")

        if request.method == 'POST':
            form = HotelForm(request.POST, request.FILES)
            img_id = request.FILES         
            if form.is_valid():
                form.save()
                if (str.__contains__(str(img_id['hotel_Main_Img']), '.mp4') ):
                    print ("it's mp4")
                    vid = FastbertConfig.ml.runModel_video(str(img_id['hotel_Main_Img']))
                    print("Video Done")
                    return render(request, 'img.html', {'form' : form, 'img_id' : img_id, 'vid' : 1})

                else:
                    print(img_id['hotel_Main_Img']) 
                    img = FastbertConfig.ml.runModel(str(img_id['hotel_Main_Img']))
                    return render(request, 'img.html', {'form' : form, 'img_id' : img_id})
                    #return redirect('success')
    
    def get(self, request):

        if request.method == 'GET':             
            form = HotelForm()        
        return render(request, 'img.html', {'form' : form})
  
  
def success(request):
    return HttpResponse('successfully uploaded')