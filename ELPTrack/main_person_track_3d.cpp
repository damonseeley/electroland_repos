// fix for some sort of max macro naming conflic with windows
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



//#include <pcl/visualization/pcl_visualizer.h>    
//#include <pcl/io/openni_grabber.h>
//#include <pcl/common/time.h>


#include <libMesaSR.h>

#include "MesaBGSubtractor.h"
#include "PointCloudConstructor.h"
#include "PlanView.h"
#include "GreedyTracker.h"
#include "Props.h"
#include "PropChangeListener.h"


#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write


using namespace pcl;
using namespace boost;
using namespace std;

boost::mutex clickMutex;

SRCAM srCam; // SwissRanger
pcl::visualization::CloudViewer* viewer;

PointCloudConstructor *cloudConstructor;
MesaBGSubtractor *bg;
PlanView *planView;
GreedyTracker *tracker;
bool removeBgFromPointCloud = false;
bool cropPointCloud = false;

//pcl::PointCloud<PointT>::Ptr transformedCloud;
//PointCloudT::Ptr clicked_points_3d;

//cv::VideoWriter outputVideo1("trackingVideoPIM1.avi", CV_FOURCC('M','P','4','2'), 20, cv::Size(60*3, 120*3), true);
//cv::VideoWriter outputVideo2("trackingVideoMJPG.avi", CV_FOURCC('M','P','4','2'), 20, cv::Size(60*3, 120*3), true);
//cv::VideoWriter outputVideo3("trackingVideoMP42.avi", CV_FOURCC('M','P','4','2'), 20, cv::Size(60*3, 120*3), true);
//cv::VideoWriter outputVideo4("trackingVideoDIV3.avi", CV_FOURCC('M','P','4','2'), 20, cv::Size(60*3, 120*3), true);
//cv::VideoWriter outputVideo5("trackingVideoDIVX.avi", CV_FOURCC('M','P','4','2'), 20, cv::Size(60*3, 120*3), true);
//cv::VideoWriter outputVideo6("trackingVideoI236.avi", CV_FOURCC('M','P','4','2'), 20, cv::Size(60*3, 120*3), true);
//cv::VideoWriter outputVideo7("trackingVideoI263.avi", CV_FOURCC('M','P','4','2'), 20, cv::Size(60*3, 120*3), true);
//cv::VideoWriter outputVideo8("trackingVideoFLV1.avi", CV_FOURCC('M','P','4','2'), 20, cv::Size(60*3, 120*3), true);
//cv::VideoWriter outputVideo9("trackingVideoUNCOMPRESSED.avi", 0, 30, cv::Size(60*3, 120*3), true);


float clickedPointX;
float clickedPointY;
float clickedPointZ;

//cv::Mat frame;

//    cv::BackgroundSubtractorMOG2 bg(1, 1, false);


struct callback_args{
	// structure used to pass arguments to the callback function
	//	PointCloudT::Ptr clicked_points_3d;
	pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};
struct callback_args cb_args;
std::string win("win"); // OpenCV window reference name 

cv::Size *imsize;


float fps;
double msPerFrame;
DWORD  lasttime;
DWORD  curtime;

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
	switch(event.getKeyCode()) {
	case 'b':
		if(event.keyUp()){ 
			removeBgFromPointCloud = ! removeBgFromPointCloud;
			std::cout << "Remove background " << removeBgFromPointCloud << std::endl;
		}
		break;
	case 'B':
		if(event.keyUp()){ 
			cropPointCloud = ! cropPointCloud;
			std::cout << "Crop Point Cloud " << cropPointCloud << std::endl;
		}
		break;

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
	bg->thresh = Props::getFloat(PROP_BG_THRESH);
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

std::cout << "    click " << current_point << std::endl;



}
*/
boolean firstFrame = true;




void aquireFrame() {

	//float fps;
	//double secsPerFrame;
	//time_t lasttime;
	//time_t curtime;
	curtime = timeGetTime();
	long elspesMS = (curtime-lasttime);
	elspesMS = elspesMS < 0 ? 0 : elspesMS; // incase of wrap around?
	elspesMS = msPerFrame - elspesMS;
	if(elspesMS > 0)
		//not sure if we need if sleep my be a no-op with negative value
		boost::this_thread::sleep( boost::posix_time::milliseconds(elspesMS) );



	SR_Acquire(srCam); // get new depth image


	imsize = new cv::Size(SR_GetCols(srCam), SR_GetRows(srCam)); // SR image size
	cv::Mat rangeImage(*imsize, CV_16UC1, SR_GetImage(srCam, 0));

	bg->process(rangeImage, removeBgFromPointCloud);


	if(bg->useAdaptive) {
		cv::Mat displayImage;
		bg->foreground.convertTo(displayImage, CV_8UC1, 256.0);
		//		imshow(win, displayImage);
	} else {
		//	imshow(win, bg->foreground);
	}
	//				cv::Mat mattedImage;
	//				mattedImage.setTo(cv::Mat::zeros);
	if(cropPointCloud) {
		cloudConstructor->aquireFrame();
		viewer->showCloud(cloudConstructor->filterFrame());
		planView->generatePlanView(cloudConstructor->filterFrame());
		tracker->updateTracks(planView->blobs,curtime);

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

		}

		imshow(win, planView->displayImage);

		/*
		outputVideo1 << planView->displayImage;
		outputVideo2 << planView->displayImage;
		outputVideo3 << planView->displayImage;
		outputVideo4 << planView->displayImage;
		outputVideo5 << planView->displayImage;
		outputVideo6 << planView->displayImage;
		outputVideo7 << planView->displayImage;
		outputVideo8 << planView->displayImage;
		*/
		//outputVideo9 << planView->displayImage;
	} else {
		viewer->showCloud(cloudConstructor->aquireFrame());
	}

	cv::waitKey(20);


	/*
	pcl::PointCloud<pcl::PointXYZ>::iterator it2 = cloud->points.begin();
	for(pcl::PointCloud<pcl::PointXYZ>::iterator it = transformedCloud->points.begin(); it != transformedCloud->points.end(); ++it,++it2) {
	std::cout << *it << "==?" << *it2 << std::endl;
	}
	*/
	lasttime = curtime;
} 

void loop() {
	while(!viewer->wasStopped())
	{
		aquireFrame();
	}

	SR_Close(srCam);

}





int main(int argc, char** argv)
{

	Props::initProps(argc, argv);
	fps = Props::getFloat(PROP_FPS);
	Props::writeToFile("testFile1.ini");
	Props::set(PROP_FPS, boost::any(10.0f));

	Props::writeToFile("testFile2.ini");


	msPerFrame = 1000.0/fps;
	curtime = timeGetTime();
	lasttime = curtime;

	SR_OpenDlg(&srCam, 2, 0); // 2 -> opens a selection dialog
	bg = new MesaBGSubtractor();
	cloudConstructor = new PointCloudConstructor(srCam);
	cloudConstructor->setWorldDims(
		Props::getFloat(PROP_MINX), Props::getFloat(PROP_MAXX),
		Props::getFloat(PROP_MINY), Props::getFloat(PROP_MAXY),
		Props::getFloat(PROP_MINZ), Props::getFloat(PROP_MAXZ));

	cloudConstructor->setTransform(
		Props::getFloat(PROP_XOFFSET), Props::getFloat(PROP_YOFFSET), Props::getFloat(PROP_ZOFFSET), 
		Props::getFloat(PROP_PITCH), Props::getFloat(PROP_YAW), Props::getFloat(PROP_ROLL)) ;

	planView = new PlanView(Props::getFloat(PROP_MINX), Props::getFloat(PROP_MAXX), Props::getFloat(PROP_MINZ), Props::getFloat(PROP_MAXZ), Props::getInt(PROP_TRACK_WIDTH), Props::getInt(PROP_TRACK_HEIGHT));

	cv::namedWindow(win); // init window
	// need to make this distance in terms of m?
	tracker = new GreedyTracker(9.0f, 1000, 1500);






	// PCL point cloud

	//transformedCloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>(SR_GetRows(srCam), SR_GetCols(srCam)));
	//		 PointCloudT::Ptr clicked_points_cloud (new PointCloudT);
	//		 clicked_points_3d = clicked_points_cloud;
	//		cb_args.clicked_points_3d = clicked_points_3d;


	viewer = new pcl::visualization::CloudViewer ("Mesa PCL example"); // visualizer

	//		cb_args.clicked_points_3d = clicked_points_3d;
	//  viewer->runOnVisualizationThreadOnce (addGroundPlane);
	viewer->runOnVisualizationThreadOnce(addBox);


	aquireFrame();
	//		viewer->registerPointPickingCallback (pp_callback, (void*)&cb_args);
	viewer->registerKeyboardCallback(kb_callback, (void*)&cb_args);

	std::cout << "shift-click on at least three points on the floor plain then press c to continue" << std::endl;

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

	loop();	

}


