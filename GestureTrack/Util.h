#ifndef __UTIL_H__
#define __UTIL_H__

#include "CinderVector.h"

#define ONE_OVER_SIXTY .0166666666666667

using namespace cinder;

class Util {
public:
	static Vec3f HSV2RGB(Vec3f hsv) {
		if(hsv.x < 0) return Vec3f(hsv.z, hsv.z, hsv.z); // assumes value between 0 and 360.  Negative means no hue
		float h = hsv.x * ONE_OVER_SIXTY;
		int i = (int) floor(h);
		float f = h - i;
		f = (!(i&1)) ?  1-f : f; // if even
		float v = hsv.z;
		float m = v * (1-hsv.y); 
		float n = v * (1-hsv.y * f); 
		switch(i) {
			case 6:
			case 0:
				return Vec3f(v,n,m);
			case 1:
				return Vec3f(n,v,m);
			case 2:
				return Vec3f(m,v,n);
			case 3:
				return Vec3f(m,n,v);
			case 4:
				return Vec3f(n,m,v);
			case 5:
				return Vec3f(v,m,n);
		}

	}
}
;
#endif