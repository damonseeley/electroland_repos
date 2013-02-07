#include "AgmtFlashBlock.h"

int* AgmtFlashBlock::flash = NULL;

#define FLASHBLOCKSIZE 350
// must be multiple of 7
AgmtFlashBlock::AgmtFlashBlock() {
	if (flash == NULL) {
		flash = new int[FLASHBLOCKSIZE + 1];
		int cnt = 0;
		for(int i = 0; i <= FLASHBLOCKSIZE - 7; i++) {
			flash[i++] = random(255);//r
			flash[i++] = random(255);//g
			flash[i++] = random(255);//b
			flash[i++] = random(255);//r2
			flash[i++] = random(255);//g2
			flash[i++] = random(255);//b2
			flash[i] = 10; // t
		}
		flash[FLASHBLOCKSIZE] = -1;
	}
}

void AgmtFlashBlock::apply(Avatar *avatar, Interpolators *interps) {
	for(int c = -1; c <= 1; c++) {
		for(int r = -1; r <= 1; r++) {
			for(int s = 0; s < 4; s++) {
				new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, c, r, s),
					flash, 1, (float) random(15));
			}
		}
	}
}
void AgmtFlashBlock::exit(int col, int row, Interpolators *interps) {
	for(int c = col -1; c <= col + 1; c++) {
		for(int r = row -1; r <= row + 1; r++) {
			BasePixel *p = Panels::thePanels->getPixel(Panels::A, c, r);
			for(int s = 0; s < 4; s++) {
				new IGeneric(interps, p->getSubPixel(s),
					flash, 1, random(15));
			}
		}
	}

}

