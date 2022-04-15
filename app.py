
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import torchvision
from PIL import Image

device = torch.device('cpu')

# transform = transforms.Compose([
    # transforms.Resize((3, 3)),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

#path_model = 'mobilenetv2.pth'
path_model = 'mobilenetv2_2.pth'
classes = ['Бутылка', 'Девочка', 'Облако']

image_in = st.file_uploader("Upload an image", type="jpg")

torch.hub.load("chenyaofo/pytorch-cifar-models",
                       "cifar100_mobilenetv2_x0_5",
                       #'cifar100_resnet20',
                       pretrained=True)

# class Normalize(nn.Module):
    # def __init__(self, mean, std):
        # super(Normalize, self).__init__()
        # self.mean = torch.tensor(mean).to(device)
        # self.std = torch.tensor(std).to(device)

    # def forward(self, input):
        # x = input / 255.0
        # x = x - self.mean
        # x = x / self.std
        # return x.permute(0, 3, 1, 2)
        
if image_in is not None:
     img = Image.open(image_in)
     img = img.resize((32, 32), Image.ANTIALIAS)
     images = []
     images.append(np.asarray(img))
     
     st.image(image_in, caption = 'Your Image to Classify:', use_column_width=True)
     
        
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
     
     
     # model = models.mobilenet_v2()
     
    
     
     # model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3, bias=True)
     # model.load_state_dict(torch.load(path_model))
     # model.eval()
     
     model = torch.load(path_model)
     
     #inimg = torchvision.datasets.ImageFolder('image', transform)
     #batch_t = torch.utils.data.DataLoader(
        #inimg, batch_size=1, shuffle=False)
     #batch_t = torch.utils.data.DataLoader(
        #transform(img), batch_size=1, shuffle=False)
     batch_t = torch.Tensor(images)
     out = model(batch_t)
     
     prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
     _, indices = torch.sort(out, descending=True)
     for idx in indices[0][:1]: 
        st.header(classes[idx])
        
     #output = out.argmax(dim=1)
     
     #st.header(classes[out])
          
     





# model = Homework_model()
# model.load_state_dict(torch.load(path_model))
# model.eval()

# batch_t = torch.unsqueeze(transform(image), 0)
# out = model(batch_t)
# output = np.argmax(out)
# st.write(classes[output])

#python -m streamlit.ilc run app_nn.py
