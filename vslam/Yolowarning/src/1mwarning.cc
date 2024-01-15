#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sstream>
#include <ctype.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <unistd.h>

using namespace std;

ros::Publisher yolore_pub;
map<string, vector<cv::Rect2i>> mmDetectMap;
vector<int> x;
vector<int> y;
std::string yolowarning;
std_msgs::String yoloSendMsg;
string coor_file, yolo_sub_topic, warning_topic;

void readcoor()
{
    //readtxt
    ifstream  fin;
    fin.open(coor_file, ios::in);
    if(!fin.is_open())
    {
        std::cerr<<"cannot open the file";
    }
    char buf[1024];

    while (fin >> buf)
    {
        string str;
        int flag = 0;
        for(int i=0 ;i<strlen(buf); i++)
        {
            if (buf[i] != ',')
            {
                str += buf[i];
            }
            else
            {
               flag = i; 
            }
        }
        x.push_back(atoi(str.substr(0,flag).c_str()));
        y.push_back(atoi(str.substr(flag,strlen(buf)-1).c_str()));
    }

}




void doMsg(const std_msgs::String::ConstPtr& msg_p)
{
    mmDetectMap.clear();
    std_msgs::String msg;
    msg.data = msg_p->data.c_str();

    string yolomsg;
    yolomsg = msg.data.c_str();
    // ROS_INFO("接收%s", yolomsg.c_str());
    char tempch;

    //对接收的数据进行解析，把每一个框的的左下角和右下角取出来
    string tempss;
    vector<string> name_rect;
    int number_left = 0;
    int number_right = 0;
    for (auto ch : yolomsg)
    {
        // cout << "ch :" << ch << endl;
        if (ch!=',')
        {
            tempss += ch;
            number_right++;
        }
        else
        {
            name_rect.push_back(tempss.substr(number_left,number_right));
            number_left = number_right;
        }
        
    }
    name_rect.push_back(tempss.substr(number_left,number_right));

    string name;
    //左上角(left,top), 右下角(right,height)
    int left;
    int top;
    int right;
    int bottom;
    int epoch = -1;
    // cout << "name_rect.size(): " << name_rect.size() << endl;
    for (int i=0; i<name_rect.size(); i++)
    {
        switch (i%5)
        {
        case 0:
            epoch += 1;
            name = name_rect[epoch*5+i%5];
            if (i>0)
            {
                cv::Rect2i DetectArea(left, top, right, bottom);
                mmDetectMap[name].push_back(DetectArea);
            }
            break;
        case 1:
            left = atoi(name_rect[epoch*5+i%5].c_str());
            break;  
        case 2:
            top = atoi(name_rect[epoch*5+i%5].c_str());
            break;
        case 3:
            right = atoi(name_rect[epoch*5+i%5].c_str());
            break;
        case 4:
            bottom = atoi(name_rect[epoch*5+i%5].c_str());
            break;
        default:
            break;
        }

    }

    cv::Rect2i DetectArea(left, top, right, bottom);
    mmDetectMap[name].push_back(DetectArea);

    int warning = 0;
    if (!mmDetectMap.empty())
    {
         for (auto vit = mmDetectMap.begin(); vit != mmDetectMap.end(); vit++)
        {
            if (vit->second.size() != 0)
            {
                for (auto area : vit->second)
                {
                    if (area.height+area.y>=y[area.x] or area.height+area.y>=y[area.x+area.width])
                    {
                        warning = 1;
                    }
                }
            }

        }
    }
    if (warning==1)
    {
        yolowarning = "Warning";
        yoloSendMsg.data = yolowarning;
        yolore_pub.publish(yoloSendMsg);
    }
}



int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    ros::init(argc,argv,"warningListener");
    ros::NodeHandle nh;

    string node_name = ros::this_node::getName();
    nh.param<std::string>(node_name + "/coordinate_file", coor_file, "file_not_set");
    nh.param<std::string>(node_name + "/yolo_sub_topic", yolo_sub_topic, "/orb_slam3/yolo_box_data");
    nh.param<std::string>(node_name + "/warning_topic", warning_topic, "/mower/warning");

    image_transport::ImageTransport it(nh);
    readcoor();
    

    yolore_pub = nh.advertise<std_msgs::String>(warning_topic,1);
    ros::Subscriber yolo_sub = nh.subscribe<std_msgs::String>(yolo_sub_topic,10,doMsg);
    ros::spin();
    return 0;

  
}

