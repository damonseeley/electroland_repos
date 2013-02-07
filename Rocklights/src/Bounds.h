#ifndef __BOUNDS_H__
#define __BOUNDS_H__

class Bounds {
	int size;
	// order must by minX, minY, maxX, maxY
	float* exclusionZones;

public:
	Bounds();
	~Bounds();
	bool isInBounds(float x, float y);
};

#endif
