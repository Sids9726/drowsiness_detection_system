import cv2

cap = cv2.VideoCapture(0)

while (cap.isOpened()):

    ret, frame = cap.read()

    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", gray)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cap.destroyAllWindows()
