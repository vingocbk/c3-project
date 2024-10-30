
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
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer)
{

  	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *>(viewer_void);
	if (event.getKeySym() == "Right" && event.keyDown()){
		cs.push_back(ControlState(0, -0.02, 0));
  	}
	else if (event.getKeySym() == "Left" && event.keyDown()){
		cs.push_back(ControlState(0, 0.02, 0)); 
  	}
  	if (event.getKeySym() == "Up" && event.keyDown()){
		cs.push_back(ControlState(0.1, 0, 0));
  	}
	else if (event.getKeySym() == "Down" && event.keyDown()){
		cs.push_back(ControlState(-0.1, 0, 0)); 
  	}
	if(event.getKeySym() == "a" && event.keyDown()){
		refresh_view = true;
	}
}

void Accuate(ControlState response, cc::Vehicle::Control& state){

	if(response.t > 0){
		if(!state.reverse){
			state.throttle = min(state.throttle+response.t, 1.0f);
		}
		else{
			state.reverse = false;
			state.throttle = min(response.t, 1.0f);
		}
	}
	else if(response.t < 0){
		response.t = -response.t;
		if(state.reverse){
			state.throttle = min(state.throttle+response.t, 1.0f);
		}
		else{
			state.reverse = true;
			state.throttle = min(response.t, 1.0f);

		}
	}
	state.steer = min( max(state.steer+response.s, -1.0f), 1.0f);
	state.brake = response.b;
}

void drawCar(Pose pose, int num, Color color, double alpha, pcl::visualization::PCLVisualizer::Ptr& viewer){

	BoxQ box;
	box.bboxTransform = Eigen::Vector3f(pose.position.x, pose.position.y, 0);
    box.bboxQuaternion = getQuaternion(pose.rotation.yaw);
    box.cube_length = 4;
    box.cube_width = 2;
    box.cube_height = 2;
	renderBox(viewer, box, num, color, alpha);
}

// Declare the function to use the ICP algorithm
Eigen::Matrix4d ICPAlgorithm(PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startPose) {
    Eigen::Matrix4d transforMatrix = Eigen::Matrix4d::Identity ();
    Eigen::Matrix4d transformInit = transform3D(startPose.rotation.yaw, startPose.rotation.pitch, startPose.rotation.roll, startPose.position.x, startPose.position.y, startPose.position.z);
    // transformSource
	PointCloudT::Ptr transformSource (new PointCloudT); 
	// transformPointCloud
    pcl::transformPointCloud (*source, *transformSource, transformInit);

    pcl::console::TicToc time;
    time.tic ();
    pcl::IterativeClosestPoint<PointT, PointT> icpAlgorithm;
	// setTransformationEpsilon
    icpAlgorithm.setTransformationEpsilon(1e-8); //0.000001
    int maxIterations = 60; //50; //25; //4; //3;
	// setMaximumIterations
    icpAlgorithm.setMaximumIterations(maxIterations);
	// setInputSource
    icpAlgorithm.setInputSource(transformSource);
	// setInputTarget
    icpAlgorithm.setInputTarget(target);

    PointCloudT::Ptr cloudICP (new PointCloudT);  // ICP output point cloud
	// align
    icpAlgorithm.align(*cloudICP);
    cout << "ICP has converged: " << icpAlgorithm.hasConverged () << " score: " << icpAlgorithm.getFitnessScore () <<  " time: " << time.toc() <<  " ms" << endl;

    if (icpAlgorithm.hasConverged ()) {
      transforMatrix = icpAlgorithm.getFinalTransformation ().cast<double>();
      transforMatrix =  transforMatrix * transformInit;
    } else 
		cout << "ERROR: ICP did not converge" << endl;
    return transforMatrix;
}

// Declare the function to use the NDT algorithm
Eigen::Matrix4d NDTAlgorithm(PointCloudT::Ptr mapCloud, PointCloudT::Ptr source, Pose startPose) {
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndtAlgorithm;
	// setTransformationEpsilon
    ndtAlgorithm.setTransformationEpsilon(1e-8);
	// setResolution
    ndtAlgorithm.setResolution(1); //1.5); //2); //0.5); //1);
	// setInputTarget
    ndtAlgorithm.setInputTarget(mapCloud);

    pcl::console::TicToc time;
    time.tic ();
    Eigen::Matrix4f transformInit = transform3D(startPose.rotation.yaw, startPose.rotation.pitch, startPose.rotation.roll, startPose.position.x, startPose.position.y, startPose.position.z).cast<float>();

    // Setting max number of registration iterations.
    int maxIterations = 60; //60; //50; //25; //15; //10; //6; //3;
	// setMaximumIterations
    ndtAlgorithm.setMaximumIterations(maxIterations);
	// setInputSource
    ndtAlgorithm.setInputSource(source);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNDT (new pcl::PointCloud<pcl::PointXYZ>);
	// align
    ndtAlgorithm.align (*cloudNDT, transformInit);
    cout << "NDT has converged: " << ndtAlgorithm.hasConverged () << " score: " << ndtAlgorithm.getFitnessScore () <<  " time: " << time.toc() <<  " ms" << endl;
    Eigen::Matrix4d transformationMatrix = ndtAlgorithm.getFinalTransformation ().cast<double>();
    return transformationMatrix;
}

int main(int argc, char *argv[]){

	bool useNDTAlgorithm = true;
	// empty for NDT, 2 for ICP
	// ./cloud_loc 2
	if(argc == 2 && strcmp(argv[1], "2") == 0) {
		useNDTAlgorithm = false;
	}
	cout << "useNDTAlgorithm " << useNDTAlgorithm << endl;
	auto client = cc::Client("localhost", 2000);
	client.SetTimeout(2s);
	auto world = client.GetWorld();

	auto blueprint_library = world.GetBlueprintLibrary();
	auto vehicles = blueprint_library->Filter("vehicle");

	auto map = world.GetMap();
	auto transform = map->GetRecommendedSpawnPoints()[1];
	auto ego_actor = world.SpawnActor((*vehicles)[12], transform);

	//Create lidar
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

	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  	viewer->setBackgroundColor (0, 0, 0);
	viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);

	auto vehicle = boost::static_pointer_cast<cc::Vehicle>(ego_actor);
	Pose pose(Point(0,0,0), Rotate(0,0,0));

	// Load map
	PointCloudT::Ptr mapCloud(new PointCloudT);
  	pcl::io::loadPCDFile("../map.pcd", *mapCloud);
  	cout << "Loaded " << mapCloud->points.size() << " data points from map.pcd" << endl;
	renderPointCloud(viewer, mapCloud, "map", Color(0,0,1)); 

	typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);
	typename pcl::PointCloud<PointT>::Ptr scanCloud (new pcl::PointCloud<PointT>);

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
		}
	});
	
	Pose poseRef(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180));
	double maxError = 0;

	bool isFirstScan = true;
	while (!viewer->wasStopped())
  	{
		while(new_scan){
			std::this_thread::sleep_for(0.1s);
			world.Tick(1s);
		}
		if(refresh_view){
			viewer->setCameraPosition(pose.position.x, pose.position.y, 60, pose.position.x+1, pose.position.y+1, 0, 0, 0, 1);
			refresh_view = false;
		}
		
		viewer->removeShape("box0");
		viewer->removeShape("boxFill0");
		Pose truePose = Pose(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180)) - poseRef;
		drawCar(truePose, 0,  Color(1,0,0), 0.7, viewer);
		double theta = truePose.rotation.yaw;
		double stheta = control.steer * pi/4 + theta;
		viewer->removeShape("steer");
		renderRay(viewer, Point(truePose.position.x+2*cos(theta), truePose.position.y+2*sin(theta),truePose.position.z),  Point(truePose.position.x+4*cos(stheta), truePose.position.y+4*sin(stheta),truePose.position.z), "steer", Color(0,1,0));


		ControlState accuate(0, 0, 1);
		if(cs.size() > 0){
			accuate = cs.back();
			cs.clear();

			Accuate(accuate, control);
			vehicle->ApplyControl(control);
		}

  		viewer->spinOnce ();
		
		if(!new_scan){
			
			new_scan = true;
			// TODO: (Filter scan using voxel filter)

			if (isFirstScan) {
				isFirstScan = false;
				pose.position = truePose.position;
				pose.rotation = truePose.rotation;
			}
			// Declare voxelGrid
			pcl::VoxelGrid<PointT> voxelGrid;
			// set input cloud
			voxelGrid.setInputCloud(scanCloud);
			// Declare filterResolution
			// double filterResolution = 0.1;
			double filterResolution = 0.5;
			// set leaf size
			voxelGrid.setLeafSize(filterResolution, filterResolution, filterResolution);
			// Declare cloud filtered
			typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);
			// Filter cloud points
			voxelGrid.filter(*cloudFiltered);


			// TODO: Find pose transform by using ICP or NDT matching
			//pose = ....
			// select matchingTransform
			Eigen::Matrix4d matchingTransform;
			if (useNDTAlgorithm)
				Eigen::Matrix4d matchingTransform = NDTAlgorithm(mapCloud, cloudFiltered, pose);
			else 
				Eigen::Matrix4d matchingTransform = ICPAlgorithm(mapCloud, cloudFiltered, pose);
			// set pose
			pose = getPose(matchingTransform);

			// TODO: Transform scan so it aligns with ego's actual pose and render that scan
			// declare correctedScanCloud
			PointCloudT::Ptr correctedScanCloud (new PointCloudT);
			
			pcl::transformPointCloud (*cloudFiltered, *correctedScanCloud, matchingTransform);


			viewer->removePointCloud("scan");
			// TODO: Change `scanCloud` below to your transformed scan
			// renderPointCloud(viewer, scanCloud, "scan", Color(1,0,0) );
			renderPointCloud(viewer, correctedScanCloud, "scan", Color(1,0,0) );

			viewer->removeAllShapes();
			drawCar(pose, 1,  Color(0,1,0), 0.35, viewer);
          
          	double poseError = sqrt( (truePose.position.x - pose.position.x) * (truePose.position.x - pose.position.x) + (truePose.position.y - pose.position.y) * (truePose.position.y - pose.position.y) );
			if(poseError > maxError)
				maxError = poseError;
			double distDriven = sqrt( (truePose.position.x) * (truePose.position.x) + (truePose.position.y) * (truePose.position.y) );
			viewer->removeShape("maxE");
			viewer->addText("Max Error: "+to_string(maxError)+" m", 200, 100, 32, 1.0, 1.0, 1.0, "maxE",0);
			viewer->removeShape("derror");
			viewer->addText("Pose error: "+to_string(poseError)+" m", 200, 150, 32, 1.0, 1.0, 1.0, "derror",0);
			viewer->removeShape("dist");
			viewer->addText("Distance: "+to_string(distDriven)+" m", 200, 200, 32, 1.0, 1.0, 1.0, "dist",0);

			if(maxError > 1.2 || distDriven >= 170.0 ){
				viewer->removeShape("eval");
			if(maxError > 1.2){
				viewer->addText("Try Again", 200, 50, 32, 1.0, 0.0, 0.0, "eval",0);
			}
			else{
				viewer->addText("Passed!", 200, 50, 32, 0.0, 1.0, 0.0, "eval",0);
			}
		}

			pclCloud.points.clear();
		}
  	}
	return 0;
}
