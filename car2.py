import  cv2

car_cascade = cv2.CascadeClassifier('cars.xml')
bus_cascade = cv2.CascadeClassifier('two_wheeler.xml')
cap =cv2.VideoCapture('4.mp4')
t = 0
c = 0


while 1:
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    car = car_cascade.detectMultiScale(gray,1.1,2)
    bus = bus_cascade.detectMultiScale(gray,1.1,2)

    for x,y,z,w in car:
        cv2.rectangle(img,(x,y),(x+z,y+w),(0,255,255),2)
        t  = t + 1
        cv2.putText(img, "Car: {}".format(int(t/70)), (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0) ,3)
        for m,n,o,p in bus:
            cv2.rectangle(img, (m, n), (m + o, n + p), (0, 255, 255), 2)
            c = c + 1
            cv2.putText(img, "two_wheeler: {}".format(c), (250,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0) ,3)


    cv2.imshow("img",img)

    if cv2.waitKey(30) == 27:
        break

cv2.destroyAllWindows()