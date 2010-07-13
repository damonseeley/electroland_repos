#ifndef __RENDER_LIST__
#define __RENDER_LIST__

#include "Drawable.h"
#include <vector>

using namespace std;


class RenderList {
	vector<Drawable*> curList;
	vector<Drawable*> nextList;
public:

	 RenderList();
	 void add(Drawable *drawable);
	 void draw(DWORD curTime);

};
#endif
