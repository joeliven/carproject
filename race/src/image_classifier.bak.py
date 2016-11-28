#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from image_utils import preprocess_image

pub = rospy.Publisher('classifier_decision', Bool, queue_size=1)

class Classifier(object):

  def __init__(self):
      self.bridge = CvBridge()
      self.encoding = 'rgb8'
      self.hasSaved = False
      rospy.Subscriber("rgb/image_rect",Image,self.classify)
      rospy.spin()
    

  def classify(self,data):
        
      image = self.convertImg(data)
      image = preprocess_image(image,dbg=False)
      #if not self.hasSaved:
      #  self.hasSaved = True
      #  np.save(self.encoding,image)
      # Run classifier
      shouldTurn = False

      pub.publish(shouldTurn)

  def convertImg(self,data):
      try:
        cv_image = self.bridge.imgmsg_to_cv2(data, self.encoding)
      except CvBridgeError as e:
        print(e)

      (rows,cols,channels) = cv_image.shape
      return np.asarray(cv_image)

if __name__ == '__main__':
    print("Classifying turn decision")
    rospy.init_node('image_classifier', anonymous=True)
    try:
      Classifier()
    except rospy.ROSInterruptException:
      pass

