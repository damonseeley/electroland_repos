#include "CoordTransform.h"


CoordTransform::CoordTransform()
{
	reset();
}

void CoordTransform::reset() {
	ax = 1.0f;
	bx = 0.0f;
	cx = 0.0f;

	ay = 0.0f;
	by = 1.0f;
	cy = 0.0f;
}

// point order clockwise from NW corner
void CoordTransform::updateTransform(
float x0, float y0, float x1, float y1, float x2, float y2, 
float xp0, float yp0, float xp1, float yp1, float xp2, float yp2) {
	setMatrix(	x0 , y0 , 1, xp0,  
				x1 , y1 , 1, xp1,
				x2 , y2 , 1, xp2);
	if(! solve()) {
		cout << "Unable to solve for x transform.  Try giggling points around" << endl;
		return;
	}

	ax = (float) m[0][3];
	bx =  (float)m[1][3];
	cx =  (float)m[2][3];

	setMatrix(	x0 , y0 , 1, yp0,  
				x1 , y1 , 1, yp1,
				x2 , y2 , 1, yp2);
	if(! solve()) {
		cout << "Unable to solve for y transform.  Try giggling points around" << endl;
		return;
	}

	ay =  (float)m[0][3];
	by =  (float)m[1][3];
	cy =  (float)m[2][3];

//	cout << ax << " " << bx << " 
	
}
void CoordTransform::setMatrix(float a0, float b0, float c0, float d0, float a1, float b1, float c1, float d1, float a2, float b2, float c2, float d2) {
	m[0][0] = a0;
	m[0][1] = b0;
	m[0][2] = c0;
	m[0][3] = d0;
	m[1][0] = a1;
	m[1][1] = b1;
	m[1][2] = c1;
	m[1][3] = d1;
	m[2][0] = a2;
	m[2][1] = b2;
	m[2][2] = c2;
	m[2][3] = d2;
}

void CoordTransform::scaleRow(int r, double f) {
	m[r][0] *= f;
	m[r][1] *= f;
	m[r][2] *= f;
	m[r][3] *= f;
}

void CoordTransform::swapRow(int a, int b) {
	double t;
	for(int i = 0; i < 4; i++) {
		t = m[a][i];
		m[a][i] = m[b][i];
		m[b][i] = t;
	}
}

bool CoordTransform::swapNonZero(int r) {
	if(m[r][r] != 0.0) return true;
	int cur = r;
	while(m[cur][r] == 0.0) {
		cur++;
		if(cur >= 3) {
			return false;
		}
	}
	swapRow(cur, r);
	return true;
}

bool CoordTransform::solve() {

	if(! swapNonZero(0)) {
		return false;
	}
	scaleRow(0, 1.0 / m[0][0]);
	
	double mult = -m[1][0];
	m[1][0] = 0.0;
	m[1][1] += (m[0][1] * mult);
	m[1][2] += (m[0][2] * mult);
	m[1][3] += (m[0][3] * mult);
	mult = -m[2][0];
	m[2][0] = 0.0;
	m[2][1] += (m[0][1] * mult);
	m[2][2] += (m[0][2] * mult);
	m[2][3] += (m[0][3] * mult);

	if(! swapNonZero(1)) {
		return false;
	}
	scaleRow(1, 1.0 / m[1][1]);
	mult = -m[0][1];
	m[0][1] += 0;
	m[0][2] += (m[1][2] * mult);
	m[0][3] += (m[1][3] * mult);
	mult = -m[2][1];
	m[2][1] += 0;
	m[2][2] += (m[1][2] * mult);
	m[2][3] += (m[1][3] * mult);
	if(! swapNonZero(2)) {
		return false;
	}
	scaleRow(1, 1.0 / m[2][2]);
	mult = -m[0][2];
	m[0][2] += 0;
	m[0][3] += (m[2][3] * mult);
	mult = -m[1][2];
	m[1][2] += 0;
	m[1][3] += (m[2][3] * mult);

	return true;

}


void CoordTransform::transform(float &x, float &y) {
	x= ax * x + bx * y + cx;
	y= ay * x  + by * y  + cy;
}

