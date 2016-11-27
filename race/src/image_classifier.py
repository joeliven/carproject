#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

pub = rospy.Publisher('classifier_decision', Bool, queue_size=1)

class Classifier(object):

  def __init__(self):
      self.bridge = CvBridge()
      self.encoding = 'bgr8'
      rospy.Subscriber("rgb/image_rect",Image,self.classify)
      rospy.spin()
    

  def classify(self,data):
        
      image = self.convertImg(data)

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

