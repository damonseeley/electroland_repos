#include "transformer.h"
#include "Profile.h"

Transformer *Transformer::theTransformer = NULL;

Transformer::Transformer(float w, float h) {
	if(theTransformer == NULL) {
		theTransformer = this;
	}
	width = w;
	height = h;


	pt[0][0] = CProfile::theProfile->Float("transP0x", 0.0f);
	pt[0][1] = CProfile::theProfile->Float("transP0y", 0.0f);
	pt[1][0] = CProfile::theProfile->Float("transP1x", width);
	pt[1][1] = CProfile::theProfile->Float("transP1y", 0.0f);
	pt[2][0] = CProfile::theProfile->Float("transP2x", width);
	pt[2][1] = CProfile::theProfile->Float("transP2y", height);
	pt[3][0] = CProfile::theProfile->Float("transP3x", 0);
	pt[3][1] = CProfile::theProfile->Float("transP3y", height);

	calc();
}
void Transformer::reset() {
	

	pt[0][0] = 0;
	pt[0][1] = 0;
	pt[1][0] = width;
	pt[1][1] = 0;
	pt[2][0] = width;
	pt[2][1] = height;
	pt[3][0] = 0;
	pt[3][1] = height;

	calc();
}
Transformer::~Transformer()
{
}
void Transformer::transform(float &x, float &y) {
	float xy = x * y;
	x = a * x + b * y+ c * xy + d;
	y = e * x + f * y+ g * xy + h;
}

//  0 1
//  3 2 
//
void Transformer::calc() {
	cout << "--------------------------------" << endl;
	cout <<  "transP0x=" << pt[0][0] << endl;
	cout <<  "transP0y=" << pt[0][1] << endl;
	cout <<  "transP1x=" << pt[1][0] << endl;
	cout <<  "transP1y=" << pt[1][1] << endl;
	cout <<  "transP2x=" << pt[2][0] << endl;
	cout <<  "transP2y=" << pt[2][1] << endl;
	cout <<  "transP3x=" << pt[3][0] << endl;
	cout <<  "transP3y=" << pt[3][1] << endl;
	cout << "--------------------------------" << endl;

	getParams(0, width, width, 0, a, b, c, d);
	getParams(0, 0, height, height, e, f, g, h);
	
}

void Transformer::getParams(float p1, float p2, float p3, float p4, float &a, float &b, float &c, float &d) {
	float x1 = pt[0][0];
	float y1 = pt[0][1];
	float x2 = pt[1][0];
	float y2 = pt[1][1];
	float x3 = pt[2][0];
	float y3 = pt[2][1];
	float x4 = pt[3][0];
	float y4 = pt[3][1];

		a = (y1*(x1*(p2*(y4 - y3) - p3*y4 + p4*y3 + (p3 - p4)*y2)
			+ p2*(x3*y3 - x4*y4) + p3*x4*y4 - p4*x3*y3 + (p4 - p3)*x2*y2)
			+ p1*(y2*(x4*y4 + x2*(y3 - y4) - x3*y3) + y3*(x3*y4 - x4*y4))
			+ y2*(x2*(p3*y4 - p4*y3) - p3*x4*y4 + p4*x3*y3) + p2*y3*(x4*y4 - x3*y4))
			/(y1*(x1*(x2*(y4 - y3) - x3*y4 + x4*y3 + (x3 - x4)*y2) + x2*(x3*y3 -
			x4*y4)
			+ x3*x4*y4 - x3*x4*y3 + x2*(x4 - x3)*y2)
			+ x1*(y2*(x4*y4 + x2*(y3 - y4) - x3*y3) + y3*(x3*y4 - x4*y4))
			+ y2*(x2*(x3*y4 - x4*y3) - x3*x4*y4 + x3*x4*y3) + x2*y3*(x4*y4 -
			x3*y4));
		b = - (p1*(x2*(x4*y4 - x3*y3) - x3*x4*y4 + x3*x4*y3 + x2*(x3 - x4)*y2)
			+ x1*(p2*(x3*y3 - x4*y4) + p3*x4*y4 - p4*x3*y3 + (p4 - p3)*x2*y2)
			+ p2*(x3*x4*y4 - x3*x4*y3) + x2*(p4*x3*y3 - p3*x4*y4) + x2*(p3*x4 -
			p4*x3)*y2
			+ x1*(p2*(x4 - x3) - p3*x4 + p4*x3 + (p3 - p4)*x2)*y1)
			/(y1*(x1*(x2*(y4 - y3) - x3*y4 + x4*y3 + (x3 - x4)*y2) + x2*(x3*y3 -
			x4*y4)
			+ x3*x4*y4 - x3*x4*y3 + x2*(x4 - x3)*y2)
			+ x1*(y2*(x4*y4 + x2*(y3 - y4) - x3*y3) + y3*(x3*y4 - x4*y4))
			+ y2*(x2*(x3*y4 - x4*y3) - x3*x4*y4 + x3*x4*y3) + x2*y3*(x4*y4 -
			x3*y4));
		c = (p1*(x2*(y4 - y3) - x3*y4 + x4*y3 + (x3 - x4)*y2) + p2*(x3*y4 -
			x4*y3)
			+ x1*(p3*y4 + p2*(y3 - y4) - p4*y3 + (p4 - p3)*y2) + x2*(p4*y3 - p3*y4)
			+ (p3*x4 - p4*x3)*y2 + (p2*(x4 - x3) - p3*x4 + p4*x3 + (p3 - p4)*x2)*y1)
			/(y1*(x1*(x2*(y4 - y3) - x3*y4 + x4*y3 + (x3 - x4)*y2) + x2*(x3*y3 -
			x4*y4)
			+ x3*x4*y4 - x3*x4*y3 + x2*(x4 - x3)*y2)
			+ x1*(y2*(x4*y4 + x2*(y3 - y4) - x3*y3) + y3*(x3*y4 - x4*y4))
			+ y2*(x2*(x3*y4 - x4*y3) - x3*x4*y4 + x3*x4*y3) + x2*y3*(x4*y4 -
			x3*y4));
		d = (p1*(y2*(x2*(x3*y4 - x4*y3) - x3*x4*y4 + x3*x4*y3) + x2*y3*(x4*y4
			- x3*y4))
			+ y1*(x1*(p2*(x4*y3 - x3*y4) + x2*(p3*y4 - p4*y3) + (p4*x3 - p3*x4)*y2)
			+ p2*(x3*x4*y4 - x3*x4*y3) + x2*(p4*x3*y3 - p3*x4*y4) + x2*(p3*x4 -
			p4*x3)*y2)
			+ x1*(y2*(x2*(p4*y3 - p3*y4) + p3*x4*y4 - p4*x3*y3) + p2*y3*(x3*y4 -
			x4*y4)))
			/(y1*(x1*(x2*(y4 - y3) - x3*y4 + x4*y3 + (x3 - x4)*y2) + x2*(x3*y3 -
			x4*y4)
			+ x3*x4*y4 - x3*x4*y3 + x2*(x4 - x3)*y2)
			+ x1*(y2*(x4*y4 + x2*(y3 - y4) - x3*y3) + y3*(x3*y4 - x4*y4))
			+ y2*(x2*(x3*y4 - x4*y3) - x3*x4*y4 + x3*x4*y3) + x2*y3*(x4*y4 -
			x3*y4));
}




