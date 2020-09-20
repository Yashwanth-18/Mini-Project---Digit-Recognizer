import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

m_new=tf.keras.models.load_model('digits_pred.h5')

img = np.ones([400,400],dtype ='uint8')*255
img[50:350,50:350]=0

wname='Canvas'
cv.namedWindow(wname)

state = False

def shape(event,x,y,flags,param):
    global state
    if event == cv.EVENT_LBUTTONDOWN:
        state = True
        cv.circle(img,(x,y),10,(255,255,255),-1)
    elif event == cv.EVENT_MOUSEMOVE:
        if state == True:
            cv.circle(img,(x,y),10,(255,255,255),-1)

    else:
        state  = False

cv.setMouseCallback(wname,shape)

while True:
    cv.imshow(wname,img)
    k = cv.waitKey(1)  
    if k == ord('q'):
        break
    elif k == ord('c'):
        img[50:350,50:350]=0
    elif k == ord('w'):
        out = img[50:350,50:350]
        cv.imwrite('sample1.jpg',img)
    elif k == ord('p'):
        #click 'w' and then press 'p' to predict digit
        out = img[50:350,50:350]
        image_test_resize = cv.resize(out ,(28,28)).reshape(1,28,28)
        dig = np.argmax(m_new.predict(image_test_resize),axis=-1)
        print("Digit recognised:",dig)


cv.destroyAllWindows()
