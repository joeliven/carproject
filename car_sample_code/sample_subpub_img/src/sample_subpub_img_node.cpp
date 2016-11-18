#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
using namespace std;


void cv_process_img(const cv::Mat& input_img, cv::Mat& output_img)
{
   cv::Mat gray_img;
   cv::cvtColor(input_img, gray_img, CV_RGB2GRAY);
   
   double t1 = 20;
   double t2 = 50;
   int apertureSize = 3;
   
   cv::Canny(gray_img, output_img, t1, t2, apertureSize); 
}

void cv_publish_img(image_transport::Publisher &pub, cv::Mat& pub_img)
{
	//sensor_msgs::ImagePtr pub_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", pub_img).toImageMsg();
	sensor_msgs::ImagePtr pub_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", pub_img).toImageMsg();
	pub.publish(pub_msg);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg, image_transport::Publisher &pub)
{
    cv_bridge::CvImageConstPtr cv_ori_img_ptr;
    try{
        cv::Mat cv_ori_img = cv_bridge::toCvShare(msg, "bgr8")->image;
        cv::Mat cv_output_img;
        cv_process_img(cv_ori_img, cv_output_img);
	cv_publish_img(pub, cv_output_img);
        //cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
        //cv::imshow("view", cv_output_img);
        //cv::waitKey(30);
    }catch(cv_bridge::Exception& e){
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_listener");

    ros::NodeHandle nh;
    
    //cv::namedWindow("view");
    //cv::startWindowThread();
    image_transport::ImageTransport it(nh);
	
ros::NodeHandle nh_pub;
image_transport::ImageTransport itpub(nh_pub);
image_transport::Publisher pub = itpub.advertise("sample/cannyimg", 1);

    image_transport::Subscriber sub = it.subscribe("rgb/image_rect", 1, boost::bind(imageCallback, _1, pub));
    ros::spin();
    //cv::destroyWindow("view");
    
}
