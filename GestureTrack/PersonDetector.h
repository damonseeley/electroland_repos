#ifndef __PERSON_DETECTOR_H__
#define __PERSON_DETECTOR_H__

#include "ProjCell.h"
#include "Projection.h"
#include "TrackHash.h"
#include "cv.h"
#include "highgui.h"

#define IMG_DISPLAY_SCALE 5
#define MAX_TRACK_DIGITS 6
#define MAX_TRACK 100000

class PersonDetector {

public:
	CvFont font;
	char trackLabel[MAX_TRACK_DIGITS];

	long curTrackID;

	long curFrame;
	long frameLife;

	TrackHash newTracks;
	TrackHash existingTracks;

	Projection *projection;
	float *heights;
	float *conv;
	float *localMax;

	float *filter;

	float minPersonArea;
	float maxPersonArea;
	float minHandDist;
	float maxMatchDistSqr;

	float imageToWorldScaleX;
	float imageToWorldScaleY;
	float imageToWorldScaleZ;
	float worldToImageScaleX;
	float worldToImageScaleY;
	float worldToImageScaleZ;


	//static float personFilter[];

	CvMemStorage* 	g_storage ;
	CvMat cvMatHeights;
	CvMat cvMatThresh;
	CvMat cvMatCont;
	CvMat cvMatDisplay;
	CvMat cvMatContMask;

	PersonDetector(Projection *projection);
	void calc(long curFrame);
	void render(float* map);
	float guassian(int x, int y, float s);
	float distSQR(CvPoint* p1, CvPoint* p2);
	void constructConv(float *a, int w, int h);
	void cpu_convole(float* d_src, float* d_dst, int w, int h, float *d_conv, int cWidth, int cHeight, bool mirrorBoarder); // cpu version of gpu code for debuggin
	void cpu_local_max(float* d_src, float* d_dst, int w, int h, int nHoodWidth, int nHoodHeight, float aveThresh) ;
	
}
;

#endif