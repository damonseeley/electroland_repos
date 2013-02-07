#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__
#include "CoordTransform.h"
#include <iostream>
using namespace std;
class Transformer
{
	
private:

	float width;
	float height;

public:
	float a,  b,  c,  d;
	float e,  f, g,  h;

	float pt[4][2];
	static Transformer *theTransformer;

public:
	Transformer(float width, float height);
	~Transformer();
	 void setPoint(int i, float x, float y) { pt[i][0] = x; pt[i][1] = y; }
	 void incPoint(int i, int d, float amt) { pt[i][d] += amt; }
	 void calc();
	 void transform(float &x, float &y);
	 void reset();

	 void getParams(float p1, float p2, float p3, float p4, float &a, float &b, float &c, float &d);
};

#endif