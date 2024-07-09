import cv2
capture = cv2.VideoCapture("out.avi")



##Might work but didn't for the videos I created


while True:
    ret, img = capture.read()

#    result = processFrame(img)
    result = img

    cv2.imshow('some', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

