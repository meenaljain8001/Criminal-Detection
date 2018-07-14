import cv2
import sqlite3
cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("myfacehaarcascade.xml")
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainer.yml")
font=cv2.FONT_HERSHEY_DUPLEX
path="dataSet"

def getProfile(id):
	con=sqlite3.connect("database")
	cmd="SELECT * FROM test WHERE ID="+str(id)
	cursor=con.execute(cmd)
	profile=None
	for row in cursor:
		profile=row
	con.close()
	return profile
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30))
		#flags = cv2.CV_HAAR_SCALE_IMAGE)

	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x ,y), (x+w, y+h), (223, 200, 31), 2)
		id,config=rec.predict(gray[y:y+h,x:x+w])
		if(config<60):
			profile=getProfile(id)
		else:
			id=0
			profile=getProfile(id)
		if profile!=None:
			cv2.rectangle(frame, (x ,y), (x+w, y+h), (223, 200, 31), 2)
			cv2.putText(frame,str(profile[1]),(x,y+h),font,1,(0,0,0))
			cv2.putText(frame,str(profile[2]),(x,y+h+30),font,1,(0,0,0))
			cv2.putText(frame,str(profile[3]),(x,y+h+60),font,1,(0,0,0))
			cv2.putText(frame,str(profile[4]),(x,y+h+90),font,1,(0,0,0))
		

	# Display the resulting frame
	cv2.imshow('Detection Required', frame)
	if cv2.waitKey(10) == 27:
		# When everything done, release the capture
		cap.release()
		cv2.destroyAllWindows()
		break
