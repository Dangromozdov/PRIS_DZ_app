
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import torchvision
from PIL import Image

device = torch.device('cpu')

path_model = 'mobilenetv2_2.pth'
classes = ['Бутылка', 'Девочка', 'Облако']

#Окно для загрузки изображения на классификацию
image_in = st.file_uploader("Upload an image", type="jpg")

#вспомогательная загрузка для модели из ДЗ
torch.hub.load("chenyaofo/pytorch-cifar-models",
                       "cifar100_mobilenetv2_x0_5",
                       pretrained=True)
       
if image_in is not None:
     img = Image.open(image_in)
     #преобразуем входные данные к виду, в котором их получает нейронная сеть в домашнем задании 
     img = img.resize((32, 32), Image.ANTIALIAS)
     images = []
     images.append(np.asarray(img))
     
     # Выводим классифицируемое изображение на экран
     st.image(image_in, caption = 'Your Image to Classify:', use_column_width=True)
     
     #В модели класс Normalize вводится отдельно, поэтому перенесём его в код приложения   
     class Normalize(nn.Module):
         def __init__(self, mean, std):
             super(Normalize, self).__init__()
             self.mean = torch.tensor(mean).to(device)
             self.std = torch.tensor(std).to(device)

         def forward(self, input):
             x = input / 255.0
             x = x - self.mean
             x = x / self.std
             return x.permute(0, 3, 1, 2)
     
     
     
     #загрузим переобученную ранее модели
     model = torch.load(path_model)
     
     #преобразуем входные данные в тензор с батчем размером 1(загруженное изображение)
     batch_t = torch.Tensor(images)
     
     #Получаем выход для модели
     out = model(batch_t)
     
     #Сортируем классы по убыванию вероятности принадлежности изображения к данному классу
     prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
     _, indices = torch.sort(out, descending=True)
     #Выводим один результат, наиболее вероятный
     for idx in indices[0][:1]: 
        st.header(classes[idx])
        
     
     






