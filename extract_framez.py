import cv2
vidcap = cv2.VideoCapture('vid.mp4')
success,image = vidcap.read()
count = 0
while success:
  if count < 90:
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
