// fix for some sort of max macro naming conflict with windows
#define NOMINMAX
#include <Windows.h> // only on a Windows system
#undef NOMINMAX

#include <string>


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

#include "MesaCam.h"
#include "MesaBGSubtractor.h"
#include "PointCloudConstructor.h"
#include "PlanView.h"
#include "GreedyTracker.h"
#include "Props.h"
#include "PropChangeListener.h"
#include "OSCTrackSender.h"
#include "ErrorLog.h"

//#include "FlyCap.h"
//#include "PulseDetector.h"


using namespace pcl;
using namespace boost;
using namespace std;

enum exitValues { 
	SUCCESS = 0, 
	GENERAL_ERROR = 1,
	AQUISITION_ERROR = 2
};

boost::mutex displayImageMutex;


MesaCam *mesaCam;
pcl::visualization::CloudViewer* viewer;
PointCloudConstructor *cloudConstructor;
MesaBGSubtractor *bg;
PlanView *planView;
GreedyTracker *tracker;
bool removeBgFromPointCloud = false;
bool cropPointCloud = false;

bool showTracks = true;
bool showRange = true;
bool showGray = true;
bool showBGSub = true;

cv::Mat displayImage;
cv::Mat rangeImage;
cv::Mat bgSubImage; 
cv::Mat grayImage;


struct callback_args{
	// structure used to pass arguments to the callback function
	//	PointCloudT::Ptr clicked_points_3d;
	pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};
struct callback_args cb_args;

std::string trackWin("Tracks"); // OpenCV window reference name 
std::string grayWin("Mesa Intensity"); // OpenCV window reference name 
std::string rangeWin("Mesa Range"); // OpenCV window reference name 
std::string bgSubWin("BG Subtraction");
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

int badFrames = 0;

int print_help()
{
	cout << "*******************************************************" << std::endl;
	cout << "Person Track 3D options:" << std::endl;
	cout << "   --help    <show_this_help>" << std::endl;
	cout << "*******************************************************" << std::endl;
	return 0;
}


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
void displayKeyboardHelp() {
	std::cout << "h   - display this help message" << std::endl; 
	std::cout << "b   - toggle background subtraction" << std::endl;
	std::cout << "B   - toggle point cloud cropping" << std::endl;
	std::cout << "S   - save current configuration to .ini file" << std::endl;
	std::cout << "1/!   - decrement/increment minx" << std::endl;
	std::cout << "2/@   - decrement/increment miny" << std::endl;
	std::cout << "3/#   - decrement/increment minz" << std::endl;
	std::cout << "4/$   - decrement/increment maxx" << std::endl;
	std::cout << "5/%   - decrement/increment maxy" << std::endl;
	std::cout << "6/^   - decrement/increment maxz" << std::endl;
	std::cout << "y/Y   - decrement/increment point count threshold in plan view" << std::endl;
	std::cout << "t/T   - decrement/increment background model threshold" << std::endl;
	std::cout << "q/Q   - quit" << std::endl;
	std::cout << "uArrow/dArrow         - pitch point cloud " << std::endl;
	std::cout << "shift lArrow/lArrow   -yaw point cloud " << std::endl;
	std::cout << "lArrow/rArrow         - roll point cloud " << std::endl;
	std::cout << "shift uArrow/dArrow   -translate point cloud in y " << std::endl;


}

int mesaCallback(SRCAM srCam, unsigned int msg, unsigned int param, void *data) {
	if(msg & CM_MSG_DISPLAY) {
		return 1;
	} else {
	std::cout << "got a call back" << std::endl;
		return 0;

	}
}

void kb_callback(const pcl::visualization::KeyboardEvent& event, void *args) {

	bool needWorldReset = true;

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
			break;
		case 'h':
			displayKeyboardHelp();
			break;
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
	case 'Y': 
		Props::inc(PROP_PLANVIEW_THRESH, 1);
		planView->pointCntThresh = Props::getInt(PROP_PLANVIEW_THRESH);
		needWorldReset = false;
		std::cout << PROP_PLANVIEW_THRESH << " " << Props::getInt(PROP_PLANVIEW_THRESH) << std::endl;
		break;

	case 'y': {
		Props::inc(PROP_PLANVIEW_THRESH, -1);
		int val = Props::getInt(PROP_PLANVIEW_THRESH);
		if(val < 0) {
			val = 0;
			Props::set(PROP_PLANVIEW_THRESH, 0);
		}
		planView->pointCntThresh = val;
		std::cout << PROP_PLANVIEW_THRESH << " " << Props::getInt(PROP_PLANVIEW_THRESH) << std::endl;
		needWorldReset = false;
			  }
		break;
	case 'T': 
		Props::inc(PROP_BG_THRESH, 0.0001f);
		bg->thresh = Props::getFloat(PROP_BG_THRESH);
		needWorldReset = false;
		break;
	case 't':
		Props::inc(PROP_BG_THRESH, -0.0001f);
		bg->thresh = Props::getFloat(PROP_BG_THRESH);
		needWorldReset = false;
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
		} else 	if(event.getKeySym() == "Left") {
			Props::inc(PROP_YAW, -.01f);
		} else 	if(event.getKeySym() == "Right") {
			Props::inc(PROP_YAW, .01f);
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

	if(needWorldReset)
		resetWorldAndCamDims();



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


	// get new depth image


	if(mesaCam->aquire()) {
		badFrames = 0;
		// getImage returns the actaul data from the camera that gets transformed into a point cloud
		// bg->process modifies the data so we call it bgSubImage here 
		// and make a copy to rangeImage if needed
		bgSubImage = mesaCam->getRangeImage();
		if(showRange)
			bgSubImage.copyTo(rangeImage);
		bg->process(bgSubImage, removeBgFromPointCloud);
		if(showGray)
			mesaCam->getIntensityImage().copyTo(grayImage);



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
	} else { // if unable to aquireFrame from mesa
		badFrames++;
		if(badFrames % 5 == 0) {
			*ErrorLog::log << "Unable to aquire " << badFrames << " images in a row from Mesa" << std::endl;
		}
		if(badFrames % 20 == 0) {
			*ErrorLog::log << "EXITING do to inability to aquire images" << std::endl;
			exit(GENERAL_ERROR);
		}
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

#
//allows for ports even though I don't think we should have them with mesa
bool isIP(const char *ipadd) {
	unsigned b1, b2, b3, b4, port = 0;
	unsigned char sep, c;
	int rc;
	rc = sscanf(ipadd, "%3u.%3u.%3u.%3u%c%u%c",
		&b1, &b2, &b3, &b4, &sep, &port, &c);
	if ( ! (rc == 4 || rc == 6)) 
		return false;
	if (rc == 6 && sep != ':') 
		return false;
	if ((b1 | b2 | b3 | b4) > 255 || port > 65535) 
		return false;
	if (strspn(ipadd, "0123456789.:") < strlen(ipadd))  
		return false;
	return true;


}


int main(int argc, char** argv)
{
	isRunning = true;	
	ErrorLog::setLogFile(DEFAULT_ERROR_LOG);
	Props::initProps(argc, argv);
	ErrorLog::setLogFile(Props::getString(PROP_ERROR_LOG));

	*ErrorLog::log << "------ Starting Up with " << Props::getString(PROP_FILE) << " ------" << std::endl;

	showTracks = Props::getBool(PROP_SHOW_TRACKS);
	showGray = Props::getBool(PROP_SHOW_GRAY);
	showRange = Props::getBool(PROP_SHOW_RANGE);
	showBGSub = Props::getBool(PROP_SHOW_BGSUB);


	fps = Props::getFloat(PROP_FPS);
	msPerFrame = 1000.0/fps;

	curtime = timeGetTime();
	lasttime = curtime;
	lastFPSEpoch = curtime;

	mesaCam = new MesaCam();
	SR_SetCallback(mesaCallback);
	mesaCam->open(Props::getString(PROP_MESA_CAM).c_str(), isIP(Props::getString(PROP_MESA_CAM).c_str()));
	if(Props::getBool(PROP_USE_MESA_CAM_SETTINGS)) {
		mesaCam->setPropsFromCam();
	} else {
		mesaCam->setupCameraFromProps();
	}

	bg = new MesaBGSubtractor();

	cloudConstructor = new PointCloudConstructor(mesaCam->srCam);
	planView = new PlanView(Props::getFloat(PROP_MINX), Props::getFloat(PROP_MAXX), Props::getFloat(PROP_MINZ), Props::getFloat(PROP_MAXZ), Props::getInt(PROP_PLANVIEW_WIDTH), Props::getInt(PROP_PLANVIEW_HEIGHT));
	planView->pointCntThresh = Props::getInt(PROP_PLANVIEW_THRESH);
	planView->setFlipX(Props::getBool(PROP_PLANVIEW_FLIPX));
	planView->setFlipZ(Props::getBool(PROP_PLANVIEW_FLIPZ));

	oscTrackSender = new OSCTrackSender(Props::getString(PROP_OSC_ADDRESS), Props::getInt(PROP_OSC_PORT));

	resetWorldAndCamDims();


	if(showTracks)
		cv::namedWindow(trackWin, 	 CV_WINDOW_NORMAL|CV_GUI_NORMAL); // init window
	if(showGray)
		mesaCam->setMode();
		cv::namedWindow(grayWin, 	 CV_WINDOW_NORMAL|CV_GUI_NORMAL); // init window
	if(showRange)
		cv::namedWindow(rangeWin, 	 CV_WINDOW_NORMAL|CV_GUI_NORMAL); // init window

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

	removeBgFromPointCloud = true;
	cropPointCloud = true;

	if(Props::getBool(PROP_SHOW_POINTS)) {
		viewer = new pcl::visualization::CloudViewer ("Mesa PCL example"); // visualizer
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

	if(showTracks || showGray || showRange || showBGSub) {
		displayImage = cv::Mat(cv::Size(10,10), CV_8UC1);
		rangeImage = cv::Mat(cv::Size(10,10), CV_8UC1);
		bgSubImage = cv::Mat(cv::Size(10,10), CV_8UC1);
		grayImage = cv::Mat(cv::Size(10,10), CV_8UC1);
		CreateThread( NULL, 0, loopThread, NULL, 0, NULL); 
		while(isRunning) {
			if(showTracks)
				cv::imshow(trackWin, displayImage);
			if(showGray)
				cv::imshow(grayWin,grayImage);
			if(showRange)
				cv::imshow(rangeWin,rangeImage);
			if(showBGSub)
				cv::imshow(bgSubWin,bgSubImage);
			cv::waitKey(30);
		}
	} else {
		loop();
	}

}


