# Handwritten-_digit_recognition
## a Handwritten digit recognition application using three classifers
### 1.Introduction
In this application, I will use three classifier: support vector machine(svm), three-layer neural network, and decistion tree to recognize handwritten digit at same time seperately and finally output the majority of these three outputs

The library I use is from the MNIST which is publicly available from http://yann.lecun.com/exdb/mnist/. Since origin format is confusing, I choose to install mnist as a python package:
```
pip install python-mnist
```
and then simply do:
```
from mnist import MNIST 
mndata = MNIST('original_data')
images, labels = mndata.load_training()
```
and we get can all images and labels of MNIST
### 2.Approach
First step I do is to preprocessing data,I rearrange the dataset into 3 files and every files contains the information of 20,000 pictures, and then I training three classifier using these 3 different dataset.

In this project, there are 2,000 picture in my input files due to the limitation of the hardware.

### 3.Result
All the output is in the ouput directory, the number classifer recognized is stored in the .csv file

I also build a ShowPic() function which can output the picture in the input file because all the picture has been transfered into the pixel value array. 


  
