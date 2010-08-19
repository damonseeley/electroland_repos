# ifndef __TYZX_CAM_H__
# define __TYZX_CAM_H__

#include "CinderMatrix.h"
#include "CinderVector.h"
#include "GestureTypeDefs.h"
#include "TyzxTCPreceiver.h"
#include "Axis.h"
#include "SyncedThreadLoop.h"

#include <string>

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

using namespace cinder;
using namespace std;


class Grabber;

class TyzxCam
{
private:
		unsigned short *zImage;

public:

	boost::barrier *barrier;
	Grabber* grabber;

	Axis *axis;

	bool isCamOn;
	
	 char *name;
	 char *camIP;

	Vec3d translation;
	Vec3d rotation;
	double scale;
	
	int port;
	TyzxTCPreceiver *receiver;

	int imgWidth;
	int imgHeight;

//	Matrix44d transformMatrix;// 4 by 4
	double tMatrix[12];
	CamParams params;


	TyzxCam(string name,  Vec3d translation,  Vec3d rotation, double scale, string camIP, int port = TCP_EXAMPLE_PORT);
	~TyzxCam();

//	void translate(Vec3d trans);
//	void rotate(Vec3d trans);
//	void applyWorldTransforms(Vec3d trans, Vec3d rot);
	void renderAxis();

	void updateTransform();

	void applyRotation(Vec3f euler);

	void setTransform(Vec3d translation, Vec3d rotation, double scale);

	void setScale(double scale) { this->scale = scale;} 
	void setMatrix(
		double m0,	double m1,	double m2,
		double m3,	double m4,	double m5,
		double m6,	double m7,	double m8,
		double m9,  double m10, double m11);

	void setMatrixInternal(
		double m0,	double m1,	double m2,
		double m3,	double m4,	double m5,
		double m6,	double m7,	double m8,
		double m9,  double m10, double m11);

	//	void setMatrix(Matrix44d m) {transformMatrix=m; }

	void transfromPoint(Vec3f &p);

	bool start();
	bool grab();


	unsigned short *getZImage();

};

class Grabber : public SyncedThreadLoop {
public:
	TyzxCam* cam;

	Grabber(boost::barrier *bar, TyzxCam *cam) :  SyncedThreadLoop(bar) {
		this->cam = cam;
	}
	
	virtual void run() {
		cam->grab();
	}
}
;



# endif
