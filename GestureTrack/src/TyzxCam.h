# ifndef __TYZX_CAM_H__
# define __TYZX_CAM_H__




#include "TyzxTCPreceiver.h"


class TyzxCam
{
public:

	float* points;
	
	
	
	const char *camIP;
	int port;
	TyzxTCPreceiver *receiver;

	unsigned short *zImage;
	
	int zWidth;
	int zHeight;

	double cx;
	double cy;
	
	double imageCenterU;
	double imageCenterV;


	TyzxCam(const char *camIP, int port = TCP_EXAMPLE_PORT);
	~TyzxCam();

	bool start();
	bool grab();

};


# endif
