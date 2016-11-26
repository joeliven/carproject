#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from race.msg import drive_param
from race.msg import pid_input

kp = 14.0
kd = 0.09
servo_offset = 18.5
prev_error = 0.0 
vel_input = 25.0

pub = rospy.Publisher('drive_parameters', drive_param, queue_size=1)

class Controller(object):
    '''
    Controller class.
    '''
    def __init__(self):
      self.classifier_decision = False
      rospy.Subscriber("error", pid_input, self.control)
      rospy.Subscriber("classifier_decision", Bool, updateClassifierDecision)
      rospy.spin()

    def updateClassifierDecision(self,data):
      self.classifier_decision = data

    def control(self,data):
	global prev_error
	global vel_input
	global kp
	global kd

	## Your code goes here
	# 1. Scale the error
	# 2. Apply the PID equation on error
	# 3. Make sure the error is within bounds
	angle = data.pid_error*kp;
	if angle > 100:
		angle = 100
	if angle < -100:
		angle = -100

	## END

        # do extra processing with the classifier decision

	msg = drive_param();
	if(data.pid_vel == 0):
		msg.velocity = -8
	else:
		msg.velocity = vel_input	
	msg.angle = angle
	pub.publish(msg)

if __name__ == '__main__':
    global kp
    global kd
    global vel_input
    print("Listening to error for PID")
    kp = input("Enter Kp Value: ")
    kd = input("Enter Kd Value: ")
    vel_input = input("Enter Velocity: ")
    rospy.init_node('pid_controller', anonymous=True)
    try:
      Controller()
    except rospy.ROSInterruptException:
      pass
