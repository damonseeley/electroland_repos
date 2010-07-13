#include "RenderList.h"

RenderList::RenderList() {
}

void RenderList::add(Drawable *drawable) {
	nextList.push_back(drawable);
}

void RenderList::draw(DWORD curTime) {
	for (vector<Drawable *>::iterator it = curList.begin(); it!=curList.end(); ++it) {
		if((*it)->draw(curTime) ){
			nextList.push_back(*it);
		} else {
			delete *it;
		}
	}
	curList.clear();
	curList.swap(nextList);
}

