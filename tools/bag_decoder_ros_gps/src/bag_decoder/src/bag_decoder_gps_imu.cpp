/*
 * coding: utf-8 -*-
 * -----------------------------------------------------------------------------
 * author: Donghee Paek, AVELab, KAIST
 * date:   2022.12.20
 * e-mail: donghee.paek@kaist.ac.kr
 * -----------------------------------------------------------------------------
 * description: script for rosbag decoder
 *				tstamp, lat, lon, alt, cov[9] for gps.txt
 *				tstamp, angular_vel_x, y, z, cov[9],
 *						linear_acc_x, y, z, cov[9] for imu.txt
 * how to: change PATH_BAG, PATH_GEN
 * release: 2022.11.21
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
// #include <tf/tf.h>

// rosbag
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Vector3.h>

#define foreach BOOST_FOREACH

using std::cout; using std::endl;
using std::system; using std::string;

// Path
static const std::string PATH_BAG = "/media/avelab/23_SSD/data_2/2022-12-09-05-44-47/2022-12-09-05-44-47.bag";
static const std::string PATH_GEN = "/media/avelab/23_SSD/gen_files";

// predefined function
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 5)
{
       std::ostringstream out;
       out << std::setprecision(n) << a_value;
       return out.str();
}

void create_directory(const string path_create)
{
	boost::filesystem::path dir(path_create);
	if(boost::filesystem::create_directory(dir)) {
		std::cout << "Cretae folder: " << path_create << std::endl;
	}
}

// (TBD) IMU Generation for piksi

// GPS Geneartion
void process_gps_with_topic(const std::string name_topic, const std::string name_file)
{
	string path_topic_folder = PATH_GEN + "/odom";
	create_directory(path_topic_folder);

	rosbag::Bag bag;
	bag.open(PATH_BAG);

	std::vector<std::string> topics;
    topics.push_back(std::string(name_topic));
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    int idx_file = 0;
	std::ostringstream ss_time_values;
	double lat, lon, alt;
	double cov_0, cov_1, cov_2, cov_3, cov_4, cov_5, cov_6, cov_7, cov_8;
    foreach(rosbag::MessageInstance const m, view)
    {
        sensor_msgs::NavSatFix::ConstPtr p_nav = m.instantiate<sensor_msgs::NavSatFix>();
        if (p_nav != NULL)
        {
			// cout << p_nav << endl;
			lat = p_nav->latitude;
			lon = p_nav->longitude;
			alt = p_nav->altitude;
			cov_0 = p_nav->position_covariance[0];
			cov_1 = p_nav->position_covariance[1];
			cov_2 = p_nav->position_covariance[2];
			cov_3 = p_nav->position_covariance[3];
			cov_4 = p_nav->position_covariance[4];
			cov_5 = p_nav->position_covariance[5];
			cov_6 = p_nav->position_covariance[6];
			cov_7 = p_nav->position_covariance[7];
			cov_8 = p_nav->position_covariance[8];

			// cout << lat << "," << lon << "," << alt << ',';
			// cout << cov_0 << "," << cov_1 << "," << cov_2 << ",";
			// cout << cov_3 << "," << cov_4 << "," << cov_5 << ",";
			// cout << cov_6 << "," << cov_7 << "," << cov_8 << "\n";

			ss_time_values << m.getTime() << ",";
			ss_time_values << lat << "," << lon << "," << alt << ',';
			ss_time_values << cov_0 << "," << cov_1 << "," << cov_2 << ",";
			ss_time_values << cov_3 << "," << cov_4 << "," << cov_5 << ",";
			ss_time_values << cov_6 << "," << cov_7 << "," << cov_8 << "\n";
        }
    }

	// save file
    string path_gps_info_file = path_topic_folder + "/" + name_file + ".txt";
    std::ofstream ss_txt_file(path_gps_info_file);
    ss_txt_file << ss_time_values.str();
    ss_txt_file.close();

	topics.clear();
}

// IMU Geneartion for os_cloud_node
void process_imu_with_topic(const std::string name_topic, const std::string name_file)
{
	string path_topic_folder = PATH_GEN + "/odom";
	create_directory(path_topic_folder);

	rosbag::Bag bag;
	bag.open(PATH_BAG);

	std::vector<std::string> topics;
    topics.push_back(std::string(name_topic));
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    int idx_file = 0;
	std::ostringstream ss_time_values;
	// Quaternion returns nan values
	// geometry_msgs::Quaternion quat;
	// double x, y, z, w;
	geometry_msgs::Vector3 vec3;
	double cov_0, cov_1, cov_2, cov_3, cov_4, cov_5, cov_6, cov_7, cov_8;
    foreach(rosbag::MessageInstance const m, view)
    {
        sensor_msgs::Imu::ConstPtr p_imu = m.instantiate<sensor_msgs::Imu>();
        if (p_imu != NULL)
        {
			// cout << p_imu << endl;
			ss_time_values << m.getTime() << ",";

			// quat = p_imu->orientation;
			// cov_0 = p_imu->orientation_covariance[0];
			// cov_1 = p_imu->orientation_covariance[1];
			// cov_2 = p_imu->orientation_covariance[2];
			// cov_3 = p_imu->orientation_covariance[3];
			// cov_4 = p_imu->orientation_covariance[4];
			// cov_5 = p_imu->orientation_covariance[5];
			// cov_6 = p_imu->orientation_covariance[6];
			// cov_7 = p_imu->orientation_covariance[7];
			// cov_8 = p_imu->orientation_covariance[8];

			// cout << x << "," << y << "," << z << ',' << w << ',';
			// cout << quat.x << "," << quat.y << "," << quat.z << "," << quat.w << ",";
			// cout << cov_0 << "," << cov_1 << "," << cov_2 << ",";
			// cout << cov_3 << "," << cov_4 << "," << cov_5 << ",";
			// cout << cov_6 << "," << cov_7 << "," << cov_8 << "\n";

			// ss_time_values << quat.x << "," << quat.y << "," << quat.z << "," << quat.w << ",";
			// ss_time_values << cov_0 << "," << cov_1 << "," << cov_2 << ",";
			// ss_time_values << cov_3 << "," << cov_4 << "," << cov_5 << ",";
			// ss_time_values << cov_6 << "," << cov_7 << "," << cov_8 << ",";

			// angular velocity
			vec3 = p_imu->angular_velocity;
			cov_0 = p_imu->angular_velocity_covariance[0];
			cov_1 = p_imu->angular_velocity_covariance[1];
			cov_2 = p_imu->angular_velocity_covariance[2];
			cov_3 = p_imu->angular_velocity_covariance[3];
			cov_4 = p_imu->angular_velocity_covariance[4];
			cov_5 = p_imu->angular_velocity_covariance[5];
			cov_6 = p_imu->angular_velocity_covariance[6];
			cov_7 = p_imu->angular_velocity_covariance[7];
			cov_8 = p_imu->angular_velocity_covariance[8];

			ss_time_values << vec3.x << "," << vec3.y << "," << vec3.z << ",";
			ss_time_values << cov_0 << "," << cov_1 << "," << cov_2 << ",";
			ss_time_values << cov_3 << "," << cov_4 << "," << cov_5 << ",";
			ss_time_values << cov_6 << "," << cov_7 << "," << cov_8 << ",";

			// linear acceleration
			vec3 = p_imu->linear_acceleration;
			cov_0 = p_imu->linear_acceleration_covariance[0];
			cov_1 = p_imu->linear_acceleration_covariance[1];
			cov_2 = p_imu->linear_acceleration_covariance[2];
			cov_3 = p_imu->linear_acceleration_covariance[3];
			cov_4 = p_imu->linear_acceleration_covariance[4];
			cov_5 = p_imu->linear_acceleration_covariance[5];
			cov_6 = p_imu->linear_acceleration_covariance[6];
			cov_7 = p_imu->linear_acceleration_covariance[7];
			cov_8 = p_imu->linear_acceleration_covariance[8];

			ss_time_values << vec3.x << "," << vec3.y << "," << vec3.z << ",";
			ss_time_values << cov_0 << "," << cov_1 << "," << cov_2 << ",";
			ss_time_values << cov_3 << "," << cov_4 << "," << cov_5 << ",";
			ss_time_values << cov_6 << "," << cov_7 << "," << cov_8 << "\n";

			cout << m.getTime() << ",";
			cout << vec3.x << "," << vec3.y << "," << vec3.z << "\n";
        }
    }

	// save file
    string path_gps_info_file = path_topic_folder + "/" + name_file + ".txt";
    std::ofstream ss_txt_file(path_gps_info_file);
    ss_txt_file << ss_time_values.str();
    ss_txt_file.close();

	topics.clear();
}

int main(int argc, char** argv)
{
	std::cout << "Decoding bag file" << std::endl;

	// create_directory(PATH_GEN);

	process_gps_with_topic("/ublox/fix", "gps");
	process_imu_with_topic("/os_cloud_node/imu", "imu");

	return 0;
}

