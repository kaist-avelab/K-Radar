/*
 * coding: utf-8 -*-
 * -----------------------------------------------------------------------------
 * author: Donghee Paek, AVELab, KAIST
 * date:   2022.01.23
 * e-mail: donghee.paek@kaist.ac.kr
 * -----------------------------------------------------------------------------
 * description: script for rosbag decoder
 * how to: change PATH_BAG, PATH_GEN
 */

// std
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <time.h>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

// rosbag
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

// pcl
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>

// opencv
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#define foreach BOOST_FOREACH

using std::cout; using std::endl;
using std::system; using std::string;

// Path
static const std::string PATH_BAG = "/home/donghee/radar_test.bag";
static const std::string PATH_GEN = "/home/donghee/gen_files";

// predefined function
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 5)
{
       std::ostringstream out;
       out << std::setprecision(n) << a_value;
       return out.str();
}

// own point type
struct OusterPointType
{
       PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding

       float intensity;
       uint16_t reflectivity;
       // float ring;
       // float noise;
       // float range;
       EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
}EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (OusterPointType,           // here we assume a XYZ + "test" (as fields)
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, intensity, intensity)
                                   (uint16_t, reflectivity, reflectivity)
                                   // (float, ring, ring)
                                   // (float, noise, noise)
                                   // (float, range, range)
);

void create_directory(const string path_create)
{
	boost::filesystem::path dir(path_create);
	if(boost::filesystem::create_directory(dir)) {
		std::cout << "Cretae folder: " << path_create << std::endl;
	}
}

void process_image_camera_with_topic(const std::string name_topic, const std::string name_file)
{
	cv_bridge::CvImage cv_img_bev, cv_img_frontal;

	// create directory
	string path_topic_folder = PATH_GEN + '/' + name_file;
	create_directory(path_topic_folder);

	rosbag::Bag bag;
	bag.open(PATH_BAG);

	std::vector<std::string> topics;
    topics.push_back(std::string(name_topic));
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    int idx_file = 0;
    foreach(rosbag::MessageInstance const m, view)
    {
        sensor_msgs::CompressedImage::ConstPtr p_img = m.instantiate<sensor_msgs::CompressedImage>();
        if (p_img != NULL)
        {
			cv_bridge::CvImagePtr p_cv_img;
			p_cv_img = cv_bridge::toCvCopy(p_img, sensor_msgs::image_encodings::BGR8);
        	std::ostringstream file_name;
			file_name << path_topic_folder << '/' << name_file << '_' << std::to_string(idx_file) << ".png";

			cout << file_name.str() << endl;
			cv::imwrite(file_name.str(), p_cv_img->image);
    		idx_file++;
        }
    }

	topics.clear();
}

void process_point_cloud2_lidar_with_topic(const std::string name_topic, const std::string name_file)
{
	// create directory
	string path_topic_folder = PATH_GEN + '/' + name_file;
	create_directory(path_topic_folder);

	rosbag::Bag bag;
	bag.open(PATH_BAG);

	std::vector<std::string> topics;
    topics.push_back(std::string(name_topic));
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    int idx_file = 0;
    foreach(rosbag::MessageInstance const m, view)
    {
        sensor_msgs::PointCloud2::ConstPtr p_point_cloud = m.instantiate<sensor_msgs::PointCloud2>();
        if (p_point_cloud != NULL)
        {
			pcl::PointCloud<OusterPointType>::Ptr p_pc_lidar(new pcl::PointCloud<OusterPointType>);
			pcl::fromROSMsg(*p_point_cloud, *p_pc_lidar);
        	std::ostringstream file_name;
			file_name << path_topic_folder << '/' << name_file << '_' << std::to_string(idx_file) << ".pcd";

			cout << file_name.str() << endl;
    		pcl::io::savePCDFileASCII(file_name.str(), *p_pc_lidar);
    		idx_file++;
        }
    }

	topics.clear();
}

void get_time_in_string_to_txt(const std::string name_topic, const std::string name_file, const std::string name_type)
{
	// create directory
	string path_time_info = PATH_GEN + '/' + "time_info";
	create_directory(path_time_info);

	rosbag::Bag bag;
	bag.open(PATH_BAG);

	std::vector<std::string> topics;
    topics.push_back(std::string(name_topic));
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    int idx_file = 0;
	std::ostringstream ss_name_file_and_time;
    foreach(rosbag::MessageInstance const m, view)
    {
    	// cout << m.getTime() << endl;
		ss_name_file_and_time << name_file << '_' << std::to_string(idx_file) << "." << name_type;
    	ss_name_file_and_time << ", " << m.getTime() << "\n";
    	idx_file++;
    }

    // cout << ss_name_file_and_time.str();
    // save file
    string path_time_info_file = path_time_info + "/" + name_file + ".txt";
    std::ofstream ss_txt_file(path_time_info_file);
    ss_txt_file << ss_name_file_and_time.str();
    ss_txt_file.close();

	topics.clear();
}


int main(int argc, char** argv)
{
	std::cout << "Decoding bag file" << std::endl;

	create_directory(PATH_GEN);

	// point cloud
	process_point_cloud2_lidar_with_topic("/os_cloud_node_1/points_1", "os1-128");
	get_time_in_string_to_txt("/os_cloud_node_1/points_1", "os1-128", "pcd");
	process_point_cloud2_lidar_with_topic("/os_cloud_node_2/points_1", "os2-64");
	get_time_in_string_to_txt("/os_cloud_node_2/points_1", "os2-64", "pcd");

	// camera
	process_image_camera_with_topic("/cam0/usb_cam/image_raw/compressed", "cam-front");
	get_time_in_string_to_txt("/cam0/usb_cam/image_raw/compressed", "cam-front", "png");
	process_image_camera_with_topic("/cam1/usb_cam/image_raw/compressed", "cam-left");
	get_time_in_string_to_txt("/cam1/usb_cam/image_raw/compressed", "cam-front", "png");

	return 0;
}

