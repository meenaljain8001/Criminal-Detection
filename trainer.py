import os
import cv2
import numpy 
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create()
path='dataSet'

def imageId(path):
	imagepath=[os.path.join(path,f) for f in os.listdir(path)]
	face=[]
	Id=[]
	for ipath in imagepath:
		faceimg=numpy.array(Image.open(ipath).convert('L'),'uint8')
		Ids=int(os.path.split(ipath)[-1].split('.')[1])
		face.append(faceimg)
		Id.append(Ids)
		cv2.imshow("trainer",faceimg)
		cv2.waitKey(100)
	return numpy.array(Id),face
ID,FACE=imageId(path)
recognizer.train(FACE,ID)
recognizer.save('recognizer/trainer.yml')
cv2.destroyAllWindows()

	
		
		
