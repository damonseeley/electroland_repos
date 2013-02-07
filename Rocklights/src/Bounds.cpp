#include "bounds.h"
#include "Profile.h"
#include "StringUtils.h"

Bounds::Bounds() {
	string str = CProfile::theProfile->String("exclusionZones", "");
	vector<string> ls;
	StringUtils::split(str, ls);
	size = ls.size();
	exclusionZones = StringUtils::createFltArray(ls);
}

bool Bounds::isInBounds(float x, float y) {
	int i = 0;
	float isInBounds = true;
	while(i < size) {
		if(x > exclusionZones[i]) {
			if(y > exclusionZones[i+1]) {
				if(x < exclusionZones[i+2]) {
					if(y < exclusionZones[i+3]) {
						return false;
					}
				}
			}
		}
		i +=4;
	}
	return true;
}

Bounds::~Bounds(void) {
	delete[] exclusionZones;
}
