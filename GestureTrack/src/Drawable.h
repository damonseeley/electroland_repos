#ifndef __DRAWABLE_H__
#define __DRAWABLE_H__

#include <windows.h>

class Drawable {
public:
	virtual bool draw(DWORD curTime) { return false;};

}
;
#endif