# caesar-yolo
Radio source detection with YOLO object detector

## **Status**
This software is under development and supported only on python 3 + tensorflow 1x. 

## **Credit**
This software is distributed with GPLv3 license. If you use it for your research, please add repository link or acknowledge authors in your papers.   

## **Installation**  

To build and install the package:    

* Download the software in a local directory, e.g. ```SRC_DIR```:   
  ```$ git clone https://github.com/SKA-INAF/caesar-yolo.git```   
* Create and activate a virtual environment, e.g. ```caesar-yolo```, under a desired path ```VENV_DIR```     
  ```$ python3 -m venv $VENV_DIR/caesar-yolo```    
  ```$ source $VENV_DIR/caesar-yolo/bin/activate```   
* Install dependencies inside venv:   
  ```(caesar-yolo)$ pip install -r $SRC_DIR/requirements.txt```   
* Build and install package in virtual env:   
  ```(caesar-yolo) python setup.py install```    
       
To use package scripts:

* Add binary directory to your ```PATH``` environment variable:   
  ``` export PATH=$PATH:$VENV_DIR/bin ```    
