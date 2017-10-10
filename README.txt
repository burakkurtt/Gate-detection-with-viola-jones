Viola&Jones gate detection training gate corner
-----------------------------------------------
This guide assumes you work with OpenCV 3.0.0. In this work, Viola&Jones algorithm is used to detect the closed gates with training gates three corner [left bottom(LB), right bottom(RB), Right top(RT)]
 
There are two part, first one is training the gates corner second is detecting gates using three corner. 

Training gate corners 
------------------------------------------------
To train gate corner with OpenCV haar clasifier we need positives and negatives images. You you can find positive and negatives images example in Gate-detection-with-viola-jones/example_images/positive_images & negative_images. This images just example to give you idea about how positives and negatives images should be. More sample can be used to obtain better result.

For training part, prepared repository by mrnugget is used. More information can be find from README.txt file at haar_cascades_train_submodule file 

for more information;
https://github.com/mrnugget/opencv-haar-classifier-training
http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html

V&J algorithm and installation
------------------------------------------------
This guide assumes that opencv 3.0.0 is installed.

1 - Clone this repository and go inside viola&jones(OpenCV)/haar_cascades_test
~$  git clone https://github.com/burakkurtt/Gate-detection-with-viola-jones.git
~$  cd Gate-detection-with-viola-jones/haar_cascades_test/

2 - create new folder named 'build'
~$  mkdir build

3 - ~$ cd build 
4 - ~$ cmake ..
5 - ~$ make 

After installation, there should be one executable files named 'corner_detect_haarcascades' at build directory. To execute file;
~$ ./corner_detect_haarcascades

6 - After train the corner you should put trained file inside of xml file and change the name of xml file in code. You should trained 3 corner with training part to use algoritm (LB, RB, RT corners).


