# Facial-recognition-with-dlib

In this project, we will explore the use of the dlib library to detect faces in an image and perform facial recognition. Our aim is to build a system that can identify individual faces in an image and match them with known individuals.

Dlib is a machine learning library that provides state-of-the-art algorithms for computer vision tasks such as face detection, facial landmark detection, and facial recognition. It also provides a number of pre-trained models that can be used to quickly and accurately perform these tasks.

Once the faces have been detected, we will use dlib's facial recognition model to compare each face to a database of known individuals. For each face, we will use only a single image data to make the comparison, making our system simple and efficient.

You can find the pre-trained models used for face and facial landmark detections and others [here](https://github.com/davisking/dlib-models), or you can check the files above.

In this example our <u>database consists of only 5 people</u> (1 photo each): **Bill Gates , Elon Musk , Me (Mohamed Amine), Toby Maguire and Jeff Bezos**.
The output results are as follows :

<p align="center">
  <img src="https://github.com/mohamedamine99/Facial-recognition-with-dlib/blob/main/output_fig.png">
</p>
