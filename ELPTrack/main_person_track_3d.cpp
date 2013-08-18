// fix for some sort of max macro naming conflict with windows
#define NOMINMAX
#include <Windows.h> // only on a Windows system
#undef NOMINMAX

#include <string>

#include <libMesaSR.h>

#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/core/operations.hpp>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <Eigen/Core>

#include "MesaBGSubtractor.h"
#include "PointCloudConstructor.h"
#include "PlanView.h"
#include "GreedyTracker.h"
#include "Props.h"
#include "PropChangeListener.h"
#include "OSCTrackSender.h"

#include "FlyCap.h"
#include "PulseDetector.h"


using namespace pcl;
using namespace boost;
using namespace std;


boost::mutex displayImageMutex;

SRCAM srCam; // SwissRanger
FlyCap *flyCap;
PulseDetector *pulseDetector;
pcl::visualization::CloudViewer* viewer;

PointCloudConstructor *cloudConstructor;
MesaBGSubtractor *bg;
PlanView *planView;
GreedyTracker *tracker;
bool removeBgFromPointCloud = false;
bool cropPointCloud = false;
bool showTracks = true;
cv::Mat displayImage;

struct callback_args{
	// structure used to pass arguments to the callback function
	//	PointCloudT::Ptr clicked_points_3d;
	pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};
struct callback_args cb_args;
std::string win("win"); // OpenCV window reference name 

cv::Size *imsize;


float fps;
float estimatedFPS;
double msPerFrame;
long frameCnt;
DWORD  lastFPSEpoch;
DWORD  lasttime;
DWORD  curtime;

OSCTrackSender *oscTrackSender;

bool isRunning;


int print_help()
{
	cout << "*******************************************************" << std::endl;
	cout << "Person Track 3D options:" << std::endl;
	cout << "   --help    <show_this_help>" << std::endl;
	cout << "*******************************************************" << std::endl;
	return 0;
}

using namespace pcl;

void addBox (pcl::visualization::PCLVisualizer& viewer) {
	viewer.addCube(
		cloudConstructor->minX, cloudConstructor->maxX,
		cloudConstructor->minY, cloudConstructor->maxY,
		cloudConstructor->minZ, cloudConstructor->maxZ,
		1,0,0);

}

void removeBox(pcl::visualization::PCLVisualizer& viewer) {
	viewer.removeShape("cube");
}


void  updateTrackerMaxMove() {
	if(! tracker) return;
	// the scale may be distoted so take the bigger of the twof
	float scaler = (planView->worldScaleX > planView->worldScaleZ) ? planView->worldScaleX : planView->worldScaleZ;
	tracker->maxDistSqr = scaler * (Props::getFloat(PROP_TRACK_MAX_MOVE) * Props::getFloat(PROP_TRACK_MAX_MOVE)) / 1000; // convert to planview space
		



}

void resetWorldAndCamDims() {
		cloudConstructor->minX = Props::getFloat(PROP_MINX);
	cloudConstructor->minY = Props::getFloat(PROP_MINY);
	cloudConstructor->minZ = Props::getFloat(PROP_MINZ);
	cloudConstructor->maxX = Props::getFloat(PROP_MAXX);
	cloudConstructor->maxY = Props::getFloat(PROP_MAXY);
	cloudConstructor->maxZ = Props::getFloat(PROP_MAXZ);

	cloudConstructor->setWorldDims(
		Props::getFloat(PROP_MINX), Props::getFloat(PROP_MAXX),
		Props::getFloat(PROP_MINY), Props::getFloat(PROP_MAXY),
		Props::getFloat(PROP_MINZ), Props::getFloat(PROP_MAXZ));
	cloudConstructor->setTransform(
		Props::getFloat(PROP_XOFFSET), Props::getFloat(PROP_YOFFSET), Props::getFloat(PROP_ZOFFSET), 
		Props::getFloat(PROP_PITCH), Props::getFloat(PROP_YAW), Props::getFloat(PROP_ROLL)) ;	
	planView->setWorldDims(
		Props::getFloat(PROP_MINX), Props::getFloat(PROP_MAXX),
		Props::getFloat(PROP_MINZ), Props::getFloat(PROP_MAXZ));
	oscTrackSender->setTransform(Props::getFloat(PROP_OSC_MINX), Props::getFloat(PROP_OSC_MAXX), Props::getFloat(PROP_OSC_MINZ), Props::getFloat(PROP_OSC_MAXZ), Props::getFloat(PROP_MINX), Props::getFloat(PROP_MAXX), Props::getFloat(PROP_MINZ), Props::getFloat(PROP_MAXZ)); 
	updateTrackerMaxMove();
}

void kb_callback(const pcl::visualization::KeyboardEvent& event, void *args) {
	//std::cout << event.getKeySym() << "  " << event.getKeyCode() << std::endl;
	/*
	if ((event.getKeyCode() == 'c') || (event.getKeyCode() == 'C')) {
	if (clicked_points_3d->points.size() >= 3) 
	{
	clickMutex.unlock();
	} else {
	std::cout << "At least 3 clicked points on the floor plain are required to continue" << std::endl;
	}
	}
	*/
	if(event.keyUp()){ 
		switch(event.getKeyCode()) {
		case 'b':
			removeBgFromPointCloud = ! removeBgFromPointCloud;
			std::cout << "Remove background " << removeBgFromPointCloud << std::endl;
			break;
		case 'B':
			cropPointCloud = ! cropPointCloud;
			std::cout << "Crop Point Cloud " << cropPointCloud << std::endl;
			break;
		case 'S':
			Props::writeToFile("ELPTrack");

		}
	}

	switch(event.getKeyCode()) {
	case '1':
		Props::inc(PROP_MINX, -0.1f);
		break;
	case '2':
		Props::inc(PROP_MINY, -0.1f);
		break;
	case '3':
		Props::inc(PROP_MINZ, -0.1f);
		break;
	case '!':
		Props::inc(PROP_MINX, 0.1f);
		break;
	case '@':
		Props::inc(PROP_MINY, 0.1f);
		break;
	case '#':
		Props::inc(PROP_MINZ, 0.1f);
		break;
	case '4':
		Props::inc(PROP_MAXX, -0.1f);
		break;
	case '5':
		Props::inc(PROP_MAXY, -0.1f);
		break;
	case '6':
		Props::inc(PROP_MAXZ, -0.1f);
		break;
	case '$':
		Props::inc(PROP_MAXX, 0.1f);
		break;
	case '%':
		Props::inc(PROP_MAXY, 0.1f);
		break;
	case '^':
		Props::inc(PROP_MAXZ, 0.1f);
		break;
	case 'T':
		Props::inc(PROP_BG_THRESH, 0.0001f);
		break;
	case 't':
		Props::inc(PROP_BG_THRESH, -0.0001f);
		break;
	case 'q':
	case 'Q':
		isRunning = false;
		break;
	}





	if(event.isShiftPressed()) {
		if(event.getKeySym() == "Up") {
			Props::inc(PROP_YOFFSET, .1f);
		} else 	if(event.getKeySym() == "Down") {
			Props::inc(PROP_YOFFSET, -.1f);
		}
	} else {

		if(event.getKeySym() == "Left") {
			Props::inc(PROP_ROLL, -.01f);
		} else 	if(event.getKeySym() == "Right") {
			Props::inc(PROP_ROLL, .01f);
		} else 	if(event.getKeySym() == "Up") {
			Props::inc(PROP_PITCH, -.01f);
		} else 	if(event.getKeySym() == "Down") {
			Props::inc(PROP_PITCH, +.01f);
		}
	}


	viewer->runOnVisualizationThreadOnce(removeBox);
	viewer->runOnVisualizationThreadOnce(addBox);

	resetWorldAndCamDims();

	bg->thresh = Props::getFloat(PROP_BG_THRESH);


	std::cout << "FPS: " << estimatedFPS << std::endl;
}



void addGroundPlane (pcl::visualization::PCLVisualizer& viewer) {
	pcl::ModelCoefficients plane_coeff;
	plane_coeff.values.resize (4);    // We need 4 values
	plane_coeff.values[0] = 0;
	plane_coeff.values[1] = 1;
	plane_coeff.values[2] = 0;
	plane_coeff.values[3] = 0;
	viewer.addPlane (plane_coeff);
}


/*
void
pp_callback (const pcl::visualization::PointPickingEvent& event, void* args)
{

event.getPoint(clickedPointX,clickedPointY,clickedPointZ);
PointT current_point;
current_point.x = clickedPointX;
current_point.y = clickedPointY;
current_point.z = clickedPointZ;

cb_args.clicked_points_3d->points.push_back(current_point);

std::cout << "    click " << current_point << std::endl; q
*/




void aquireFrame() {
	//float fps;
	//double secsPerFrame;
	//time_t lasttime;
	//time_t curtime;


	frameCnt++;
	curtime = timeGetTime();
	if(frameCnt % 100 == 0) {
		estimatedFPS = 100000 / (curtime-lastFPSEpoch); // (100 frames/elapsed ms * 1000 ms / 1s)
		lastFPSEpoch = curtime;
	}
	long elspesMS = (curtime-lasttime);
	elspesMS = elspesMS < 0 ? 0 : elspesMS; // incase of wrap around?
	elspesMS = msPerFrame - elspesMS;
	if(elspesMS > 0)
		//cv::waitKey(elspesMS);
		//not sure if we need if sleep my be a no-op with negative value
		boost::this_thread::sleep( boost::posix_time::milliseconds(elspesMS) );


	SR_Acquire(srCam); // get new depth image


	imsize = new cv::Size(SR_GetCols(srCam), SR_GetRows(srCam)); // SR image size
	cv::Mat rangeImage(*imsize, CV_16UC1, SR_GetImage(srCam, 0));

	bg->process(rangeImage, removeBgFromPointCloud);



	if(bg->useAdaptive) {
		bg->foreground.convertTo(displayImage, CV_8UC1, 256.0);
		//		imshow(win, displayImage);
	} else {
		//	imshow(win, bg->foreground);
	}
	//				cv::Mat mattedImage;
	//				mattedImage.setTo(cv::Mat::zeros);



	if(cropPointCloud) {
		cloudConstructor->aquireFrame();

		cloudConstructor->filterFrame();

		if(viewer)
			viewer->showCloud(cloudConstructor->filteredPtr);
		planView->generatePlanView(cloudConstructor->filteredPtr);
		tracker->updateTracks(planView->blobs, curtime, lasttime);

		oscTrackSender->sendTracks(tracker);
		//view tracks
		if(showTracks) {
			for(std::vector<Track*>::iterator trackIt = tracker->tracks.begin(); trackIt != tracker->tracks.end(); ++trackIt) {
				Track* t = *trackIt;
				int r = (t->id * 13) % 255;
				int g = (t->id * 101) % 255;
				int b = (t->id * 3) % 255;
				int col;
				int row;
				int width = (t->isProvisional) ? 2 : -1;
				int radius = (t->isMatched) ? 4 : 2;
				// big if match, solid if not provisional;		
				planView->worldDimsToBinDims(t->x, t->z, col, row);
				cv::circle(planView->displayImage, cv::Point(col*3, row*3), radius*2, CV_RGB  (r,g,b),width*2); 
				stringstream fpsMsg;
				fpsMsg << "fps: " << estimatedFPS;
				cv::putText(planView->displayImage, fpsMsg.str(), cvPoint(0, planView->displayImage.rows), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,0,0));
				displayImage = cv::Mat(planView->displayImage);
			}
		}


	} else {
		if(viewer)
			viewer->showCloud(cloudConstructor->aquireFrame());
	}



	/*r
	pcl::PointCloud<pcl::PointXYZ>::iterator it2 = cloud->points.begin();
	for(pcl::PointCloud<pcl::PointXYZ>::iterator it = transformedCloud->points.begin(); it != transformedCloud->points.end(); ++it,++it2) {
	std::cout << *it << "==?" << *it2 << std::endl;
	}
	*/
	lasttime = curtime;
	if(viewer) 
		isRunning = ! viewer->wasStopped();
} 

void loop() {
	while(isRunning)
	{
		aquireFrame();
	}

	SR_Close(srCam);

}
DWORD WINAPI loopThread( LPVOID lpParam ) {
	loop();
	return 0;
}


int mouseX=0;
int mouseY=0;

void mouseEvent(int evt, int x, int y, int flags, void* param) {                    
	mouseX = x;
	mouseY = y;
//    cv::Mat* rgb = (cv::Mat*) param;
	/*
	
		*/
}

int main(int argc, char** argv)
{
	isRunning = true;	
	Props::initProps(argc, argv);

	/* fly tests */
	cv::namedWindow(win); // init window



	showTracks = Props::getBool(PROP_SHOW_TRACKS);

	fps = Props::getFloat(PROP_FPS);

	//	FlyCap *fc  = new FlyCap();

	msPerFrame = 1000.0/fps;
	SR_OpenDlg(&srCam, 2, 0); // 2 -> opens a selection dialog
	curtime = timeGetTime();
	lasttime = curtime;
	lastFPSEpoch = curtime;
	bg = new MesaBGSubtractor();
	
	cloudConstructor = new PointCloudConstructor(srCam);
	planView = new PlanView(Props::getFloat(PROP_MINX), Props::getFloat(PROP_MAXX), Props::getFloat(PROP_MINZ), Props::getFloat(PROP_MAXZ), Props::getInt(PROP_TRACK_WIDTH), Props::getInt(PROP_TRACK_HEIGHT));
	oscTrackSender = new OSCTrackSender(Props::getString(PROP_OSC_ADDRESS), Props::getInt(PROP_OSC_PORT));
	
	resetWorldAndCamDims();
	
	
	if(showTracks)
		cv::namedWindow(win, 	 CV_WINDOW_NORMAL|CV_GUI_NORMAL); // init window
	// need to make this distance in terms of m?




	tracker = new GreedyTracker(0);
	updateTrackerMaxMove();

	Track::provisionalTime = Props::getInt(PROP_TRACK_PROVISIONAL_TIME);
	Track::provisionalTimeToDeath = Props::getInt(PROP_TRACK_TIME_TO_DIE_PROV);
	Track::timeToDeath = Props::getInt(PROP_TRACK_TIME_TO_DIE);
	Track::setSmoothing(Props::getFloat(PROP_TRACK_SMOOTHING));





	// PCL point cloud

	//transformedCloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>(SR_GetRows(srCam), SR_GetCols(srCam)));
	//		 PointCloudT::Ptr clicked_points_cloud (new PointCloudT);
	//		 clicked_points_3d = clicked_points_cloud;
	//		cb_args.clicked_points_3d = clicked_points_3d;

	if(Props::getBool(PROP_SHOW_POINTS)) {
		viewer = new pcl::visualization::CloudViewer ("Mesa PCL example"); // visualizer
	} else {
		//assume pointcloud is already setup
		removeBgFromPointCloud = true;
		cropPointCloud = true;

	}

//	aquireFrame(); // need to 
//	isRunning = true;

	//		viewer->registerPointPickingCallback (pp_callback, (void*)&cb_args);
	if(viewer) {
		viewer->runOnVisualizationThreadOnce(addBox);
		viewer->registerKeyboardCallback(kb_callback, (void*)&cb_args);
	}


	/*
	clickMutex.lock();
	clickMutex.lock();

	Eigen::VectorXf ground_coeffs;
	ground_coeffs.resize(4);
	std::vector<int> clicked_points_indices;
	for (unsigned int i = 0; i < clicked_points_3d->points.size(); i++)
	clicked_points_indices.push_back(i);

	pcl::SampleConsensusModelPlane<PointT> model_plane(clicked_points_3d);
	model_plane.computeModelCoefficients(clicked_points_indices,ground_coeffs);
	std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;
	Eigen::Vector3f planeNormal(ground_coeffs(0), ground_coeffs(1), ground_coeffs(2));
	Eigen::Vector3f upVector(0,1,0);

	Eigen::Vector3f leftInPlane = planeNormal.cross(upVector);
	floorTransform = pcl::getTransFromUnitVectorsXY(leftInPlane, planeNormal);
	*/
	
	if(showTracks) {
		displayImage = cv::Mat(cv::Size(10,10), CV_8UC1);
		CreateThread( NULL, 0, loopThread, NULL, 0, NULL); 
		while(isRunning) {
		cv::imshow(win, displayImage);
		cv::waitKey(30);
		}
	} else {
		loop();
	}

}


