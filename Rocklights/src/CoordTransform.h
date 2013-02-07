#ifndef __COORDTRANSFORM_H__
#define __COORDTRANSFORM_H__
#include <iostream>
using namespace std;

#define X 0
#define Y 1

class CoordTransform {



	float ax;
	float bx;
	float cx;

	float ay;
	float by;
	float cy;

	float width, height;

	double m[3][4];
	


public:
	CoordTransform();
	~CoordTransform() {};

	void reset();
	void updateTransform(float x0, float y0, float x1, float y1, float x2, float y2,
		float xp0, float yp0, float xp1, float yp1, float xp2, float yp2
		);
	
void setMatrix(float a0, float b0, float c0, float d0,
				float a1, float b1, float c1, float d1,
				float a2, float b2, float c2, float d2);

void scaleRow(int r, double f);
void swapRow(int a, int b);
bool swapNonZero(int r);


	bool solve();
	void transform(float &x, float &y);

//	void incPt(int p, int d, float amt) { pt[p][d] += amt;}

	
};

#endif