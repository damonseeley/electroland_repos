#include "Pattern.h"

Pattern *Pattern::theCurPattern = NULL;
Pattern *Pattern::blankPattern = NULL;
bool Pattern::inited = false;

Pattern::Pattern() {
	if(blankPattern == NULL) {
//		inited = true;
		blankPattern =this;
	}
}
void Pattern::setAvatars(PersonStats *ps, bool isEnter) {}
