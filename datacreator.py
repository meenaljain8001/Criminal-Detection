import cv2
import sqlite3

cap = cv2.VideoCapture(0)
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("myfacehaarcascade.xml")

def wish(Id,Name,Age,Gender,CriminalRecord):
	con=sqlite3.connect("database")
	cmd="SELECT * FROM test WHERE ID="+str(Id)
	cursor=con.execute(cmd)
	isRecordExist=0
	args=(Id,Name,Age,Gender,CriminalRecord)
	for row in cursor:
		isRecordExist=1
	if isRecordExist==1:
		cmd="UPDATE test (ID,Name,Age,Gender,CriminalRecord) Values('%d','%s','%d','%s','%s')"
	else:
		cmd="INSERT INTO test(ID,Name,Age,Gender,CriminalRecord) Values('%d','%s','%d','%s','%s')"
	con.execute(cmd%args)
	con.commit()
	con.close()

id=input('enter user id ')
name=input("enter user name ")
age=input("enter user age ")
gender=input("enter user gender ")
criminalrecord=input("enter user criminal record ")
wish(id,name,age,gender,criminalrecord)
sampleNum=0
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30))
	#flags = cv2.CV_HAAR_SCALE_IMAGE

	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces

	for (x, y, w, h) in faces:
		sampleNum=sampleNum+1
		cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
		cv2.rectangle(frame, (x ,y), (x+w, y+h), (223, 200, 31), 2)
		cv2.waitKey(200)
	


	# Display the resulting frame
	cv2.imshow('Detection Required', frame)
	cv2.waitKey(30)
	if sampleNum>50:
		break
	# When everything done, release the capture
		
cap.release()
cv2.destroyAllWindows()
