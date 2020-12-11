# Face Mask Detection

## Introduction

COVID-19 is a contagious disease caused due to a virus, which appears to spread from one person to another person through close contact. This outbreak in today’s world has very disastrous effects, which may sometimes lead to the death of a person. To protect oneself and save the lives of others, it is necessary to put on a mask in public places. To avoid the spread of disease in public places, a two-phase deep learning face detector model is built to detect face masks. The first phase consists of training the neural model using the image dataset and the second phase will be capturing real-time video from the camera and then detecting if the person is wearing a mask or not. 	

## Related Work

Author Jason et al, proposed a dual-stage CNN architecture for the face mask detection [1].  The first stage of architecture detects humans face using RetinaFace model, which resulted better when compared to Dlib DNN and MTCNN models. The second stage classifies faces as masked or unmasked using NASNetMobile model after comparing its performance with DenseNet121 and MobileNetV2. The system exhibited greater performance and has ability to identify multiple face masks in the image. This system can further be extended to port to machine learning models to its TensorFlow Lite versions. In the experiments conducted by Toshanlal et al. [2] in generating accurate face segmentation masks from any size of input image, predefined training weights of VGG-16 architecture is used to extract the features and make predictions. This fully convolution networks (FCN) consist of 17 convolutional layers and 5 max pooling layers. The system shows the identified faces withing bounding circle with respect to pixel level accuracy. Also, refined predicted masks are shown after they are subjected to post processing. The proposed FCN model separates facial spatial location along with a particular label. The post processing for detecting facial masks given greater boost to a mean pixel level accuracy. This system can identify both non frontal and multiple faces from single image, which can find its advanced implications in detecting facial part. 

A hybrid deep transfer learning model with machine learning models is built to detect face mask [3]. The first part is designed using Resnet50 for feature extraction. The second component is designed using classical machine learning classifiers such as decision trees, Support Vector Machine (SVM), and ensemble learning for classifying the face masks as with mask or without mask. Three algorithms are used to make comparison and find the best suitable algorithm with highest accuracy and consumed less time in training and detection processes. The authors achieved the highest accuracy with the least time consumed in the training process with SVM classifier. The SVM classifier in Real-World Masked Face Dataset (RMFD) resulted higher testing accuracy. In Simulated Masked Face Dataset (SMFD), classifier obtained comparatively less, while in Labeled Faces in the Wild (LFW) Dataset, it resulted the highest testing accuracy. In future, neutrosophic domain can used in classifying and detecting tasks. In [4], Principal Component Analysis (PCA) is implemented for facial feature extraction in detecting masked and non-masked faces. The study showed face without mask gave better recognition rate, while for masked face gave poor. It is concluded that feature extraction using PCA for masked face is not effective than non-masked face.

## Selected Dataset

Face Mask Dataset is taken from the Kaggle site. The dataset consists of images of people with mask or without mask. It has around 12,000 images. The images are grouped into test, train, and validation folders. Each folder contains images which are further grouped into WithMask and WithoutMask.

Dataset can be found [**here**.](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset)

Sample Dataset: 

![](Sample%20Dataset.png)

## Convolutional Neural Network (CNN)

Convolutional Neural Network (CNN) is a deep learning algorithm which is applied to visualize the images. CNNs are more of a regularized class of multilayer perceptrons. CNNs are specialized type of neural networks that employs convolution operations instead of matrix multiplication at least in one CNN’s layers. CNN consists of input, output, and hidden layer. The hidden layer consists of convolutional layer that convolves either multiplication or dot product. The input image given to CNN is assigned weights and biases to different aspects of input, based on which network can make differentiations one from another. The major role of ConvNet is to reduce the image size into a shape that is easy to process, without the loss of image critical features for making better prediction. Every neuron in one layer is connected to every neuron in another layer in a fully connected layer. CNN is basically designed to work with two-dimensional image data, although it can be used with one- and three-dimensional images.

## OpenCV

OpenCV (Open Source Computer Vision) is a computer vision and machine learning library which is freely available. It is built to offer basic computer Vision applications and to elevate the usage of machine learning in commercial outcomes. It is written in C++ language and its interface is in C++. OpenCV is used for the analysis of images and videos like reading license plate, face recognition and identification, advanced robotic vision, editing photo and many more. This library runs on various operating systems like Linux, OpenBSD, macOS, Windows, NetBSD, FreeBSD.

## System Architecture

### Model-1:

![](Model%201.png)

In the first phase, Model-1 is trained on face mask image dataset using a single convolutional neural layer. The image data is preprocessed by transforming it to tensors and then performed normalization before feeding neural model. The processed data passed to convolutional layer creates feature map that summarizes detected features in the input. BatchNorm2D calculates mean and variance of input features and does normalization. Further, outputs are passed through Rectified Linear Unit (ReLU) activation function. It is the most used activation function. ReLU helps neural model to learn faster and perform better by overcoming the problem of vanishing gradient. This function allows model to account for non-linearities and interactions. And then it is passed through MaxPool2D layer. This layer takes the maximum value over window defined with pool size for individual dimension along with the feature axis. Next, output of MaxPool2D layer is parsed through fully connected neural layer consisting of flatten, linear and ReLU layer. This layer flattens and applies weights outputting probabilities that determines whether the input image is WithMask or WithoutMask. 

In the second phase, the trained neural model is then deployed to capture real-time video and make analysis using OpenCV which is a computer vision and machine learning library. Here, the video is analyzed to detect if a person is wearing mask or not.

### Model-2:

![](Model%202.png)

Model-2 follows the same training and real-time video detection steps as Model-1. Model-2 consists of four convolutional layers, four ReLU layers, two max pooling layers, and a fully connected neural layer. This model is built without BatchNorm2D layer.

### Model-3:

![](Model%203.png)

Similarly, Model-3 has the same procedural steps as Model-1 and Model-2. However, it is built with six convolutional layers, six BatchNorm2D layers, two ReLU layers, two max pooling layers and a fully connected neural layer.

## Best Performing Model

## Other Models

## Results and Conclusion

## References

[1] Chavda, Amit & Dsouza, Jason & Badgujar, Sumeet & Damani, Ankit. (2020). Multi-Stage CNN Architecture for Face Mask Detection.

[2] T. Meenpal, A. Balakrishnan and A. Verma, "Facial Mask Detection using Semantic Segmentation," 2019 4th International Conference on Computing, Communications and Security (ICCCS), Rome, Italy, 2019, pp. 1-5, doi: 10.1109/CCCS.2019.8888092.

[3] Loey, M., Manogaran, G., Taha, M., & Khalifa, N. (2021). A hybrid deep transfer learning model with machine learning methods for face mask detection in the era of the COVID-19 pandemic. Measurement : journal of the International Measurement Confederation, 167, 108288. https://doi.org/10.1016/j.measurement.2020.108288

[4] M. S. Ejaz, M. R. Islam, M. Sifatullah and A. Sarker, "Implementation of Principal Component Analysis on Masked and Non-masked Face Recognition," 2019 1st International Conference on Advances in Science, Engineering and Robotics Technology (ICASERT), Dhaka, Bangladesh, 2019, pp. 1-5, doi: 10.1109/ICASERT.2019.8934543.
