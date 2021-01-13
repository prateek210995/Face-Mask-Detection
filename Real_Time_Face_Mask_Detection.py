#Project : Face Mask Detection Using Deep Learning

#Authors: Shivani Ghatge, Prateek Chitpur, Vinaya Chinti

#Course Project Professor’s Name : Vadim Sokolov

#Course Name: OR-610 Deep Learning: Predictive Analytics

import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from math import ceil as r 
from torchvision import transforms

#Function to calculate the accuracy
def calc_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

#Defining class for image classification
class Img_Classification(nn.Module):
    def train_step(self, batch):
        images, labels = batch 
        images=images.to(device)
        labels=labels.to(device)
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels.long()) # Calculate loss
        return loss
    
    def valid_step(self, batch):
        images, labels = batch 
        images=images.to(device)
        labels=labels.to(device)
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels.long())   # Calculate loss
        acc = calc_accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def valid_epoch(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epochstop(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
#model with one convolutional layer       
class cnn_maskdetect_convBatch(Img_Classification):
    def __init__(self):
        super().__init__()

        self.network_convBatch = nn.Sequential(  
            #Inuput_shape = [64 , 3, 100 , 100] 
            nn.Conv2d(in_channels = 3, out_channels = 100, kernel_size=3, padding=1, stride = 1),
            nn.BatchNorm2d(100),  #shape = [64 , 100, 100 , 100]
            nn.ReLU(),            #shape = [64 , 100, 100 , 100]
            nn.MaxPool2d(2, 2),   #shape = [64 , 100, 50 , 50]
            
            nn.Flatten(), 
            nn.Linear(100 * 50 * 50, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2))
        
    def forward(self, inputs):
        return self.network_convBatch(inputs)        
#model with 6 convolutional layers
class cnn_maskdetect_6convBatch(Img_Classification):
    def _init_(self):
        super()._init_()
        self.network_6convBatch = nn.Sequential(
            # input_shape = [64 , 3, 100 , 100]
            nn.Conv2d(3, 10, kernel_size=3, padding=1), #shape = [64 , 10, 100 , 100]
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1), #shape = [64 , 32, 100 , 100]
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), #shape = [64 , 64, 100 , 100]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #shape = [64 , 64, 50 , 50]
            
            nn.Conv2d(64, 100, kernel_size=3, stride=1, padding=1), #shape = [64 , 100, 100 , 100]
            nn.BatchNorm2d(100),
            nn.Conv2d(100, 128, kernel_size=3, padding=1), #shape = [64 , 128, 100 , 100]
            nn.BatchNorm2d(128), 
            nn.Conv2d(128, 200, kernel_size=3, stride=1, padding=1), #shape = [64 , 200, 50 , 50]
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #shape = [64 , 200, 25 , 25]
            
            nn.Flatten(), 
            nn.Linear(200 * 25 * 25, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 2)
        
        )
        
    def forward(self, inputs):
        return self.network_6convBatch(inputs)
#modelwith 4 convolutional layers
class cnn_maskdetect_4c2m(Img_Classification):
    def _init_(self):
        super()._init_()
        self.network_4c2m = nn.Sequential(
            # input_shape = [64,3,100,100]
            nn.Conv2d(3,100, kernel_size=3, padding=1), # shape = [64,100,100,100]
            nn.ReLU(),
            nn.Conv2d(100, 128, kernel_size=3, stride=1, padding=1), # shape = [64,128,100,100]
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # shape = [64,128,50,50]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # shape = [64,256,50,50]
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # shape = [64,256,50,50]
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # shape = [64,256,25,25]

            nn.Flatten(), 
            nn.Linear(256 * 25 * 25, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2))
        
    def forward(self, inputs):
        return self.network_4c2m(inputs)
#class to transform the image and convert to array 
class Data_preprocessing():

    def __init__(self, path=None, img=None):
        self.directory = path
        self.img = img
    
    def image_transform(self):
        transform = transforms.Compose([
                transforms.Resize((100,100)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return transform
    
    def array_transform(self):
        img = self.img
        transforms =  self.image_transform()
        return torch.tensor(np.expand_dims(transforms(img),0))

device = torch.device('cpu')

#loading the saved models 
#uncomment the model that needs to be used for testing
model1 = torch.load('C:/Users/HP/Downloads/Trained Models/Trained Models/MaskDetectModel1.pth',map_location=device)
#model2 = torch.load('C:/Users/HP/Downloads/Trained Models/Trained Models/MaskDetectModel2.pth',map_location=device)
#model3 = torch.load('C:/Users/HP/Downloads/Trained Models/Trained Models/MaskDetectModel3.pth',map_location=device)

mtcnn = MTCNN(select_largest=False, device=device)

cam = cv.VideoCapture(0)

#checking if the camera is opened 
if not (cam.isOpened()):
    print('Camera not opened')

labels = {
    1:'You are wearing mask. Thank you!',
    0:'Please wear mask for safety.'
    }
color_dict={
    0:(255,0,0),
    1:(0,128,0)
    }

#Function to detetct the face real time and predict if the person is wearing mask or not 
def detect_mask(model):
    while True:
        is_open, frame = cam.read()
        if is_open:
            img_pad =  cv.copyMakeBorder(frame, 50,50,50,50, cv.BORDER_CONSTANT)
            img_read = Image.fromarray(img_pad)
            img_cord,_ = mtcnn.detect(img_read)
            
            if img_cord is not None:
                for cord in img_cord:
                    for x1,y1,x2,y2 in [cord]:
                        x1,y1,x2,y2 = r(x1),r(y1),r(x2),r(y2)
                        img_detect = img_pad[y1:y2 ,x1:x2]
                        preprocess = Data_preprocessing(img=Image.fromarray(img_detect))
                        img_tensor = preprocess.array_transform()
                        probs, lbl = torch.max(torch.exp(model(img_tensor.to(device))),dim=1)  #making predictions 
                        
                        mes = round((y2-y1)*35/100)
                        cv.rectangle(frame, (x1-50,y1-50), (x1-40,y1-40), color_dict[lbl.item()],-1)
    
                        cv.rectangle(frame, (x1-50,y1-50), (x2-50,y2-50), color_dict[lbl.item()],1)
                
                        cv.putText(frame,labels[lbl.item()], 
                                    (x1-50,y1-53),cv.FONT_HERSHEY_SIMPLEX, mes*0.01,(255,255,0),1)
                        
            cv.imshow("Frame", frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('End')
            break
        
    cam.release()
    cv.destroyAllWindows()
#calling the detect_mask function with the model loaded previously as the argument 
#Note: Only one model can be tested at a time
detect_mask(model1)
#detect_mask(model2)
#detect_mask(model3)