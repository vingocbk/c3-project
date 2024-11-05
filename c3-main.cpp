
#include <carla/client/Client.h>
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Map.h>
#include <carla/geom/Location.h>
#include <carla/geom/Transform.h>
#include <carla/client/Sensor.h>
#include <carla/sensor/data/LidarMeasurement.h>
#include <thread>

#include <carla/client/Vehicle.h>

//pcl code
//#include "render/render.h"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace std;

#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include "helper.h"
#include <sstream>
#include <chrono> 
#include <ctime> 
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/console/time.h>   // TicToc

PointCloudT pclCloud;
cc::Vehicle::Control control;
std::chrono::time_point<std::chrono::system_clock> currentTime;
vector<ControlState> cs;

bool refresh_view = false;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *viewer)
{

	// boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *>(viewer_void);
	if (event.getKeySym() == "Right" && event.keyDown())
	{
		cs.push_back(ControlState(0, -0.02, 0));
	}
	else if (event.getKeySym() == "Left" && event.keyDown())
	{
		cs.push_back(ControlState(0, 0.02, 0));
	}
	if (event.getKeySym() == "Up" && event.keyDown())
	{
		cs.push_back(ControlState(0.1, 0, 0));
	}
	else if (event.getKeySym() == "Down" && event.keyDown())
	{
		cs.push_back(ControlState(-0.1, 0, 0));
	}
	if (event.getKeySym() == "a" && event.keyDown())
	{
		refresh_view = true;
	}
}

void Accuate(ControlState response, cc::Vehicle::Control &state)
{

	if (response.t > 0)
	{
		if (!state.reverse)
		{
			state.throttle = min(state.throttle + response.t, 1.0f);
		}
		else
		{
			state.reverse = false;
			state.throttle = min(response.t, 1.0f);
		}
	}
	else if (response.t < 0)
	{
		response.t = -response.t;
		if (state.reverse)
		{
			state.throttle = min(state.throttle + response.t, 1.0f);
		}
		else
		{
			state.reverse = true;
			state.throttle = min(response.t, 1.0f);
		}
	}
	state.steer = min(max(state.steer + response.s, -1.0f), 1.0f);
	state.brake = response.b;
}

void drawCar(Pose pose, int num, Color color, double alpha, pcl::visualization::PCLVisualizer::Ptr &viewer)
{

	BoxQ box;
	box.bboxTransform = Eigen::Vector3f(pose.position.x, pose.position.y, 0);
	box.bboxQuaternion = getQuaternion(pose.rotation.yaw);
	box.cube_length = 4;
	box.cube_width = 2;
	box.cube_height = 2;
	renderBox(viewer, box, num, color, alpha);
}

Eigen::Matrix4d ICP(PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startingPose, int iterations)
{
	Eigen::Matrix4d transformationMatrix = Eigen::Matrix4d::Identity();

	Eigen::Matrix4d initTransform = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll,
												startingPose.position.x, startingPose.position.y, startingPose.position.z)
										.cast<double>();

	PointCloudT::Ptr transformSource(new PointCloudT);
	// transformPointCloud
	pcl::transformPointCloud(*source, *transformSource, initTransform);

	pcl::console::TicToc time;
	time.tic();
	pcl::IterativeClosestPoint<PointT, PointT> icpAlgorithm;
	// setMaximumIterations
	icpAlgorithm.setMaximumIterations(iterations);
	// setInputSource
	icpAlgorithm.setInputSource(transformSource);
	// setInputTarget
	icpAlgorithm.setInputTarget(target);
	// setMaxCorrespondenceDistance
	icpAlgorithm.setMaxCorrespondenceDistance(5);
	// setTransformationEpsilon
	icpAlgorithm.setTransformationEpsilon(1e-4);
	// setEuclideanFitnessEpsilon
	icpAlgorithm.setEuclideanFitnessEpsilon(2);
	// setRANSACOutlierRejectionThreshold
	icpAlgorithm.setRANSACOutlierRejectionThreshold(0.2);

	PointCloudT::Ptr cloud_icp(new PointCloudT); // ICP output point cloud
	// align
	icpAlgorithm.align(*cloud_icp);

	if (icpAlgorithm.hasConverged())
	{
		std::cout << "\nICP has converged in " << time.toc() << "ms" << std::endl;
		std::cout << "Score: " << icpAlgorithm.getFitnessScore() << std::endl;
		transformationMatrix = icpAlgorithm.getFinalTransformation().cast<double>();
		transformationMatrix = transformationMatrix * initTransform;
		// std::cout << "transformationMatrix:\n" << transformationMatrix << std::endl;
		return transformationMatrix;
	}
	else
	{
		std::cout << "WARNING: ICP did not converge" << std::endl;
		return transformationMatrix;
	}
}

Eigen::Matrix4d NDT(PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startingPose, int iterations)
{
	// Initialise NDT
	pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndtAlgorithm;
	// setTransformationEpsilon
	ndtAlgorithm.setTransformationEpsilon(1e-4);
	// setStepSize
	ndtAlgorithm.setStepSize(0.005);
	// setResolution
	ndtAlgorithm.setResolution(0.5);
	// setInputTarget
	ndtAlgorithm.setInputTarget(target);

	Eigen::Matrix4d transformationMatrix = Eigen::Matrix4d::Identity();

	Eigen::Matrix4d initTransform = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll,
												startingPose.position.x, startingPose.position.y, startingPose.position.z)
										.cast<double>();

	pcl::console::TicToc time;
	time.tic();
	
	// Setting max number of registration iterations.
	ndtAlgorithm.setMaximumIterations(iterations);
	// setInputSource
	ndtAlgorithm.setInputSource(source);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNDT(new pcl::PointCloud<pcl::PointXYZ>);
	ndtAlgorithm.align(*cloudNDT, initTransform.cast<float>());

	if (ndtAlgorithm.hasConverged())
	{
		std::cout << "\nNDT has converged in " << time.toc() << "ms" << std::endl;
		std::cout << "Score: " << ndtAlgorithm.getFitnessScore() << std::endl;
		transformationMatrix = ndtAlgorithm.getFinalTransformation().cast<double>();
		// transformationMatrix = transformationMatrix * initTransform;
		// cout << "transformationMatrix:\n" << transformationMatrix << endl;
		return transformationMatrix;
	}
	else
	{
		std::cout << "WARNING: NDT did not converge" << std::endl;
		return transformationMatrix;
	}
}

int main(int argc, char *argv[])
{
	// declare useNdtAlgorithm for choosing algorithm
	bool useNdtAlgorithm = true;
	// example "./cloud_loc" for NDT
	// example "./cloud_loc 2" for ICP
	if(argc == 2 && strcmp(argv[1], "2") == 0) useNdtAlgorithm = false;
	auto client = cc::Client("localhost", 2000);
	client.SetTimeout(50s);
	auto world = client.GetWorld();

	auto blueprint_library = world.GetBlueprintLibrary();
	auto vehicles = blueprint_library->Filter("vehicle");

	auto map = world.GetMap();
	auto transform = map->GetRecommendedSpawnPoints()[1];
	auto ego_actor = world.SpawnActor((*vehicles)[12], transform);

	// Create lidar
	auto lidar_bp = *(blueprint_library->Find("sensor.lidar.ray_cast"));
	// CANDO: Can modify lidar values to get different scan resolutions
	lidar_bp.SetAttribute("upper_fov", "15");
	lidar_bp.SetAttribute("lower_fov", "-25");
	lidar_bp.SetAttribute("channels", "32");
	lidar_bp.SetAttribute("range", "30");
	lidar_bp.SetAttribute("rotation_frequency", "60");
	lidar_bp.SetAttribute("points_per_second", "500000");

	auto user_offset = cg::Location(0, 0, 0);
	auto lidar_transform = cg::Transform(cg::Location(-0.5, 0, 1.8) + user_offset);
	auto lidar_actor = world.SpawnActor(lidar_bp, lidar_transform, ego_actor.get());
	auto lidar = boost::static_pointer_cast<cc::Sensor>(lidar_actor);
	bool new_scan = true;
	std::chrono::time_point<std::chrono::system_clock> lastScanTime, startTime;

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->registerKeyboardCallback(keyboardEventOccurred, (void *)&viewer);

	auto vehicle = boost::static_pointer_cast<cc::Vehicle>(ego_actor);
	Pose pose(Point(0, 0, 0), Rotate(0, 0, 0));

	// Load map
	PointCloudT::Ptr mapCloud(new PointCloudT);
	pcl::io::loadPCDFile("map.pcd", *mapCloud);
	cout << "Loaded " << mapCloud->points.size() << " data points from map.pcd" << endl;
	renderPointCloud(viewer, mapCloud, "map", Color(0, 0, 1));

	typename pcl::PointCloud<PointT>::Ptr cloudFiltered(new pcl::PointCloud<PointT>);
	typename pcl::PointCloud<PointT>::Ptr scanCloud(new pcl::PointCloud<PointT>);

	lidar->Listen([&new_scan, &lastScanTime, &scanCloud](auto data){

		if(new_scan){
			auto scan = boost::static_pointer_cast<csd::LidarMeasurement>(data);
			for (auto detection : *scan){
				if((detection.x*detection.x + detection.y*detection.y + detection.z*detection.z) > 8.0){
					pclCloud.points.push_back(PointT(detection.x, detection.y, detection.z));
				}
			}
			if(pclCloud.points.size() > 5000){ // CANDO: Can modify this value to get different scan resolutions
				lastScanTime = std::chrono::system_clock::now();
				*scanCloud = pclCloud;
				new_scan = false;
			}
		} });

	Pose poseRef(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi / 180, vehicle->GetTransform().rotation.pitch * pi / 180, vehicle->GetTransform().rotation.roll * pi / 180));
	double maxError = 0;

	while (!viewer->wasStopped())
	{
		while (new_scan)
		{
			std::this_thread::sleep_for(0.1s);
			world.Tick(1s);
		}
		if (refresh_view)
		{
			viewer->setCameraPosition(pose.position.x, pose.position.y, 60, pose.position.x + 1, pose.position.y + 1, 0, 0, 0, 1);
			refresh_view = false;
		}

		viewer->removeShape("box0");
		viewer->removeShape("boxFill0");
		Pose truePose = Pose(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi / 180, vehicle->GetTransform().rotation.pitch * pi / 180, vehicle->GetTransform().rotation.roll * pi / 180)) - poseRef;
		drawCar(truePose, 0, Color(1, 0, 0), 0.7, viewer);
		double theta = truePose.rotation.yaw;
		double stheta = control.steer * pi / 4 + theta;
		viewer->removeShape("steer");
		renderRay(viewer, Point(truePose.position.x + 2 * cos(theta), truePose.position.y + 2 * sin(theta), truePose.position.z), Point(truePose.position.x + 4 * cos(stheta), truePose.position.y + 4 * sin(stheta), truePose.position.z), "steer", Color(0, 1, 0));

		ControlState accuate(0, 0, 1);
		if (cs.size() > 0)
		{
			accuate = cs.back();
			cs.clear();

			Accuate(accuate, control);
			vehicle->ApplyControl(control);
		}

		viewer->spinOnce();

		if (!new_scan)
		{
			pose.position = truePose.position;
			pose.rotation = truePose.rotation;
			
			new_scan = true;
			// TODO: (Filter scan using voxel filter)
			pcl::VoxelGrid<PointT> voxelGrid;
			// setInputCloud
			voxelGrid.setInputCloud(scanCloud);
			// setLeafSize
			voxelGrid.setLeafSize(0.1f, 0.1f, 0.1f);
			// filter
			voxelGrid.filter(*cloudFiltered);

			// TODO: Find pose transform by using ICP or NDT matching
			Eigen::Matrix4d transformEstimate;
			if (useNdtAlgorithm) 
				transformEstimate = NDT(mapCloud, cloudFiltered, pose, 100); // TODO: change the number of iterations to positive number
			else
				transformEstimate = ICP(mapCloud, cloudFiltered, pose, 100); // TODO: change the number of iterations to positive number
			// TODO: Transform scan so it aligns with ego's actual pose and render that scan
			// Convert Eigen matrix to CARLA Transform
			pose = getPose(transformEstimate);
			PointCloudT::Ptr transformedScan(new PointCloudT);
			pcl::transformPointCloud(*cloudFiltered, *transformedScan, transformEstimate);

			viewer->removePointCloud("scan");
			// TODO: Change `scanCloud` below to your transformed scan
			renderPointCloud(viewer, transformedScan, "scan", Color(1, 0, 0));

			viewer->removeAllShapes();
			drawCar(pose, 1, Color(0, 1, 0), 0.35, viewer);

			double poseError = sqrt((truePose.position.x - pose.position.x) * (truePose.position.x - pose.position.x) + (truePose.position.y - pose.position.y) * (truePose.position.y - pose.position.y));
			if (poseError > maxError)
				maxError = poseError;
			double distDriven = sqrt((truePose.position.x) * (truePose.position.x) + (truePose.position.y) * (truePose.position.y));
			viewer->removeShape("maxE");
			viewer->addText("Max Error: " + to_string(maxError) + " m", 200, 100, 32, 1.0, 1.0, 1.0, "maxE", 0);
			viewer->removeShape("derror");
			viewer->addText("Pose error: " + to_string(poseError) + " m", 200, 150, 32, 1.0, 1.0, 1.0, "derror", 0);
			viewer->removeShape("dist");
			viewer->addText("Distance: " + to_string(distDriven) + " m", 200, 200, 32, 1.0, 1.0, 1.0, "dist", 0);

			if (maxError > 1.2 || distDriven >= 170.0)
			{
				viewer->removeShape("eval");
				if (maxError > 1.2)
				{
					viewer->addText("Try Again", 200, 50, 32, 1.0, 0.0, 0.0, "eval", 0);
				}
				else
				{
					viewer->addText("Passed!", 200, 50, 32, 0.0, 1.0, 0.0, "eval", 0);
				}
			}

			pclCloud.points.clear();
		}
	}
	return 0;
}