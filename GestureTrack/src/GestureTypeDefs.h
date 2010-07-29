
#ifndef __GUESTURE_TYPE_DEFS_H__
#define __GUESTURE_TYPE_DEFS_H__



#define CAM_PARAM_CNT 10
#define PI 3.14159265f

typedef struct {
	double cx;
	double cy;

	double imageCenterU;
	double imageCenterV;

	double tx;
	double ty;
	double tz;

	double rx;
	double ry;
	double rz;

} CamParams;

#endif
