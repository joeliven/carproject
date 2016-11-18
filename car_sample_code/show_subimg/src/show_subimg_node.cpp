#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>



void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImageConstPtr cv_ori_img_ptr;
    try{
        //cv::Mat cv_ori_img = cv_bridge::toCvShare(msg, "bgr8")->image;
	cv::Mat cv_ori_img = cv_bridge::toCvShare(msg, "mono8")->image;
        
        cv::imshow("view", cv_ori_img);
        cv::waitKey(30);
    }catch(cv_bridge::Exception& e){
        ROS_ERROR("Could not get image from '%s' to 'mono8'.", msg->encoding.c_str());
    }
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_show");

    ros::NodeHandle nh;
    
    cv::namedWindow("view");
    cv::startWindowThread();
    image_transport::ImageTransport it(nh);

    //image_transport::Subscriber sub = it.subscribe("rgb/image_rect", 1, imageCallback); 
    image_transport::Subscriber sub = it.subscribe("sample/cannyimg", 1, imageCallback);
    ros::spin();
    cv::destroyWindow("view");
    
}
