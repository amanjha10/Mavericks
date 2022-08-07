from django.shortcuts import render
import pickle
import  tensorflow 
import numpy as np
from numpy.linalg import norm
from   keras.api._v2.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from keras.api._v2.keras.preprocessing import image
from keras.api._v2.keras import Sequential
from keras.api._v2.keras.layers import GlobalMaxPooling2D
from .form import ImageForm
import os
import cv2
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import default_storage



def home(request):
 return render(request, 'app/home.html')

def product_detail(request):
 return render(request, 'app/productdetail.html')

def add_to_cart(request):
 return render(request, 'app/addtocart.html')

def buy_now(request):
 return render(request, 'app/buynow.html')

def profile(request):
 return render(request, 'app/profile.html')

def address(request):
 return render(request, 'app/address.html')

def orders(request):
 return render(request, 'app/orders.html')

def change_password(request):
 return render(request, 'app/changepassword.html')

def mobile(request):
 return render(request, 'app/mobile.html')

def login(request):
 return render(request, 'app/login.html')

def customerregistration(request):
 return render(request, 'app/customerregistration.html')

def checkout(request):
 return render(request, 'app/checkout.html')

def index(request):
    if request.method =="POST" and request.FILES['upload']:
        if'upload' not in request.FILES:
            err='No images Selected'
            return render(request,'base.html',{'err':err})
        f=request.FILES['upload']
        if f=='':
            wee='No files selected'
            return render(request,'base.html',{'err':err})
        upload =request.FILES['upload']
        
        #file_url=


        # image = load_img(file_url, target_size=(224, 224))
        # numpy_array = img_to_array(image)
        # image_batch = np.expand_dims(numpy_array, axis=0)
        # processed_image =ResNet50,.preprocess_input(image_batch.copy())

def fetch(request):
    if request.method =="POST": 
        print(request.FILES)
        if'upload' not in request.FILES:
            err='No images Selected'
            return render(request,'./app/final.html',{'err':err})
        upload =request.FILES['upload']
        filename =upload.name
        print(filename)
        with default_storage.open(filename, 'wb+') as destination:
            for chunk in upload.chunks():
                destination.write(chunk)
    # if request.method == 'POST' and request.FILES['upload']:
    #     upload = request.FILES['upload']
    #     fss = FileSystemStorage()
    #     file = fss.save(upload.name, upload)
    #     file_url = fss.url(file)
    #     print(file_url)
        feature_list = np.array(pickle.load(open('./shoppinglyx/embeddings.pkl','rb')))
        filenames = pickle.load(open('./shoppinglyx/filenames.pkl','rb'))
        print('ashdsajhgdasgd')
        
        model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
        model.trainable = False

        model = Sequential([
            model,
            GlobalMaxPooling2D()
        ])
        img = image.load_img(os.path.realpath(destination.name),target_size=(224,224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
        neighbors.fit(feature_list)
        distances,indices = neighbors.kneighbors([normalized_result])
        print(distances, indices)
        result_urls=[]
        for file in indices[0][0:5]:
            image_name=filenames[file].split("/")[-1]
            print(filenames[file])
            print('ashdsajhgdasgd')
            #temp_img = plt.imread(filenames[file])
            #img_url='../fashion-dataset/images'

            result_urls.append(image_name)
        print(result_urls)
        return render(request,'./app/final.html', {'context': result_urls})


def final(request):
    return render(request,'./app/final.html')