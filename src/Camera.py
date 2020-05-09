import cv2 as cv
import numpy as np

class Camera():

   def frameCapter():
      while True:
         frame = cv.VideoCapture(0).read()

         cv.imshow("frame", frame)

         key = cv.waitKey(100)
         if key == 27:
            break
         if key == ord("s"):
            return frame
      self.cap.release()
      cv.destroyAllWindows()
