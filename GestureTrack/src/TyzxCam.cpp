#include <iostream>
#include <sstream>
#include <math.h>

#include "TyzxCam.h"
#include "Tyzx3x3Matrix.h"


 
TyzxCam::TyzxCam(string namestr,  Vec3d translation,  Vec3d rotation, double scale, string camIPstr, int port) {


	barrier = new boost::barrier(2);
	grabber = new Grabber(barrier, this);


	this->scale = scale;
	axis = new Axis(translation, rotation);

	isCamOn = false;

	this->name = new char[namestr.size() + 1];
	strcpy(name, namestr.c_str());

	this->rotation = rotation;
	this->translation = translation;
	this->camIP = new char[camIPstr.size() + 1];
	strcpy(camIP, camIPstr.c_str());
	this->port = port;

	params.tx = translation.x;
	params.ty = translation.y;	
	params.tz = translation.z;
	params.rx = rotation.x;
	params.ry = rotation.y;
	params.rz = rotation.z;
	
	this->imgHeight = 0;
	this->imgWidth = 0;
	setTransform(translation, rotation, scale);

	receiver = new TyzxTCPreceiver(camIP, port);

}

bool TyzxCam::start() {
	
	if(! receiver) {
		std::cerr << "TyzxCam::start()  -- TyzxTCPreceiver could not be created for " << camIP << " on port " << port << "\n";
		return false;
	} else {
		std::cout << "TyzxCam::start()  -- Initilizing Settings for " << camIP << " on port " << port << "\n";
	}

	receiver->initializeSettings();
	
	if(! receiver->probeConfiguration()) {
		std::cerr << "TyzxCam::start()  -- probeConfiguration failed for " << camIP << " on port " << port << "\n";
		return false;
	}
	int size = receiver->zHeight() * receiver->zWidth();
//	points = new float[size*2*3];

	if (! receiver->startCapture()) {
		std::cerr << "TyzxCam::start()  -- startCapture failed for " << camIP << " on port " << port << "\n";
		return false;
	}

	isCamOn = true;


	grabber->start();
	barrier->wait(); // wait for inital grab

	return true;


}

//	boost::mutex grabbingMutex;
//	boost::mutex zImageMutex;

bool TyzxCam::grab() {
	
	if(! 	isCamOn) return false;


	if(! receiver->grab()) {
		std::cerr << "TyzxCam::grab()  -- grab failed for " << camIP << " on port " << port << "\n";
		return false;
	}

	if(! receiver->isZEnabled()) {
		std::cerr << "TyzxCam::grab()  -- grab failed for " << camIP << " on port " << port << " range imaging was not enabled"<< "\n" ;
		return false;
	}

	imgWidth = receiver->zWidth();
	imgHeight = receiver->zHeight(); 


	params.cx = receiver->getCx();
	params.cy = receiver->getCy();
	receiver->getZimageCenter(params.imageCenterU, params.imageCenterV);

	zImage = receiver->getZImage();




	return true;
	// waits automatically when done

}

unsigned short *TyzxCam::getZImage() {
	unsigned short *retVal = zImage;
	barrier->wait();
	// should return immediatly because grab should have run already
	// but should allow loop to run again
	return retVal;
}



TyzxCam::~TyzxCam() {
	receiver->closeConnection();
	delete receiver;
}	

void TyzxCam::applyRotation(Vec3f euler) {
	double toRad = PI / 180.0f;
	Tyzx3x3Matrix eu = Tyzx3x3Matrix( rotation.x * toRad, rotation.y * toRad, rotation.z * toRad);
	Tyzx3x3Matrix oldRot = Tyzx3x3Matrix(tMatrix[0],tMatrix[1],tMatrix[2],tMatrix[3],tMatrix[4],tMatrix[5],tMatrix[6],tMatrix[7],tMatrix[8]);

	Tyzx3x3Matrix newRot;
	oldRot.mult(eu, newRot);
		// rotcomp = rot1 * rot2

	
	Tyzx3Vector trans2 = Tyzx3Vector(0,0,0);
	Tyzx3Vector oldTrans = Tyzx3Vector(tMatrix[9],tMatrix[10],tMatrix[11]);
	Tyzx3Vector transComp = Tyzx3Vector();



	// transcomp = rot1*trans2 + trans1
	trans2.rigidTransform(oldRot, oldTrans, transComp);
	double *trans = transComp.array();
	double *rot = newRot.array();
	
	setTransform(Vec3f(trans[0],trans[1],trans[2]), Vec3f(rot[0],rot[1],rot[2]), scale);


}


void TyzxCam::setTransform(Vec3d translation, Vec3d rotation, double scale) {
	this->translation = translation;
	this->rotation = rotation;
	double toRad = PI / 180.0f;

	Tyzx3x3Matrix scalem = Tyzx3x3Matrix();
	scalem.identity();
	scalem.scaleMatrix(Tyzx3Vector(scale,scale,scale));
	Tyzx3x3Matrix mat = scalem * Tyzx3x3Matrix( rotation.x * toRad, rotation.y * toRad, rotation.z * toRad);
	double *a =mat.array();
	setMatrixInternal(  a[0], a[1], a[2],
				a[3], a[4], a[5],
				a[6], a[7], a[8],
				translation.x * scale, translation.y* scale, translation.z* scale);
	
	std::cout << camIP << " set from euler   T(" << translation << ")    R(" << rotation << ")" << std::endl;

}

void TyzxCam::updateTransform() {
	setTransform(translation, rotation, scale);
}

//	std::cerr << "TyzxCam " <<  name << " (" << camIP << ":" << port << ") T"<< translation  << "  R" << rotation<<std::endl ;
//	transformMatrix.setToIdentity();
//	rotation *= PI / 180.0f;
//	translation.y *= -1;
//	 transformMatrix.translate(translation);
//	 transformMatrix.rotate(rotation);
	// fix order
//	transformMatrix.rotate(Vec3f(1,0,0), rotation.x);
//	transformMatrix.rotate(Vec3f(0.0, 1,0.0), rotation.y);
//	transformMatrix.rotate(Vec3f(0.0, 0.0, 1.0), rotation.z);

//	transformMatrix.rotate(Vec3f(0.0, 0.0, 1.0), rotation.x);
//	transformMatrix.rotate(Vec3f(0.0, 1,0.0), rotation.y);
//	transformMatrix.rotate(Vec3f(0.0, 0.0, 1.0), rotation.z);

//	transformMatrix.rotate(Vec3f(0.0, 0.0, 1.0), rotation.z);

//	axis->pos = translation;
//	axis->rot = rotation;

//	std::cout << transformMatrix << std::endl;

//}	
void TyzxCam::setMatrix(
		double m0,	double m1,	double m2,	double m3,
		double m4,	double m5,	double m6,	double m7,
		double m8,	double m9,	double m10,	double m11 ) {
			
			setMatrixInternal(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11);

			double x1,y1,z1;
			double x2,y2,z2;

			Tyzx3x3Matrix(m0,m1,m2,m3,m4,m5,m6,m7,m8).findEulerAnglesXYZ(x1,y1,z1,x2,y2,z2);
			float toDeg = 180.0f/ PI;

			rotation.x = x1 * toDeg;
			rotation.y = y1 * toDeg;
			rotation.z = z1 * toDeg;
			translation.x = m9;
			translation.y = m10;
			translation.z = m11;

			std::cout << camIP << "  set from matrix   T(" << translation << ")    R(" << rotation << ")" << std::endl;
			
}
void TyzxCam::transfromPoint(Vec3f &p) {
	float x = p.x;
		float y = p.y;
		float z = p.z;

p.x =  tMatrix[0] * x	+  tMatrix[3] * y	+ tMatrix[6] * ((double) z) + tMatrix[9];	
p.y =  tMatrix[1] * x	+  tMatrix[4] * y	+ tMatrix[7] * ((double) z) + tMatrix[10];	
p.z =  tMatrix[2] * x	+  tMatrix[5] * y	+ tMatrix[8] * ((double) z) + tMatrix[11];

}

void TyzxCam::setMatrixInternal(
		double m0,	double m1,	double m2,	double m3,
		double m4,	double m5,	double m6,	double m7,
		double m8,	double m9,	double m10,	double m11 ) {
			tMatrix[0] = m0;
			tMatrix[1] = m1;
			tMatrix[2] = m2;
			tMatrix[3] = m3;
			tMatrix[4] = m4;
			tMatrix[5] = m5;
			tMatrix[6] = m6;
			tMatrix[7] = m7;
			tMatrix[8] = m8;
			tMatrix[9] = m9;
			tMatrix[10] = m10;
			tMatrix[11] = m11;
}
//void TyzxCam::applyWorldTransforms(Vec3d trans, Vec3d rot) {
//	setTransform(translation, rotation);
//	transformMatrix.translate(trans);
//	transformMatrix.rotate(rot);
//}


//void TyzxCam::translate(Vec3d trans) {
//	translation += trans;
//	params.tx = translation.x;
//	params.ty = translation.y;	
//	params.tz = translation.z;
//	setTransform(translation, rotation);
//}



//void TyzxCam::rotate(Vec3d rot) {
//
//	rotation += rot;
//	params.rx = rotation.x;
//	params.ry = rotation.y;
//	params.rz = rotation.z;	
//	setTransform(translation, rotation);
//}


void TyzxCam::renderAxis() {

}