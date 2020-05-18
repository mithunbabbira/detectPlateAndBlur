import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('car_plate.jpg')


plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')


def detect_blur_plate(img):

	plate_img = img.copy()
	# Region Of Interest
	roi = img.copy()

	# detect number plate (return the dimensions)
	plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3, minNeighbors= 3)


	for (x,y,w,h) in plate_rects:

		# image slicing from y to y+h and from x to x+w
		roi = roi[y:y+h,x:x+w]

		#blur the cropeed image
		blurred_roi = cv2.medianBlur(roi,7) 


		#overlap the blurred img on the original image
		plate_img[y:y+h,x:x+w]= blurred_roi


	cv2.imshow('test 2',plate_img)
	return plate_img





def detect_number_plate(img):

	#below code is to detect numberPlate

    plate_img = img.copy()
    
    
    plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3, minNeighbors=3)
    
    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img,(x,y),(x+w,y+h),(0,0,255),5)

    cv2.imshow('test',plate_img)




print('press "z" few times, it might take some time to open ')
print('caps off')






while True:
	detect_blur_plate(img) 
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if cv2.waitKey(1) & 0xFF == ord('z'):
		detect_number_plate(img)      

cv2.destroyAllWindows()