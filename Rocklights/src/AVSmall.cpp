/*
 *  AVPerson.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/30/05.
 * 
 *
 */

#include "AVSmall.h"

int red[] = {
    255,  -255,    -255,    255, -255, -255, 1000, 
  -1};

AVSmall::AVSmall(PersonStats *personStats, Interpolators *interps) :Avatar () {

  new IGeneric(interps, addOffsetPixel(A, 0, 0, true), red, -1);

  
}


void AVSmall::updateFrame(PersonStats *personStats, int ct, int dt, Interpolators *interps) {
	int col = personStats->col;
	int row = personStats->row;

	int pil = -1;
	int subRow = 0;
	if (row == 0) {
		if ((col == 4) || (col == 5))  {
			pil = J;
		} else if ((col == 12) || (col == 13)) {
			pil = I;
		} else if ((col == 17) || (col == 16)) {
			pil = G;
		}
	} else if (row == 1) {
		if ((col == 18) || (col == 19)) {
			pil = H;
		}
	} else if (row == 10) {
		if ((col == 18) || (col == 19)) {
			pil = E;
		}
	} else if (row == 11) {
		if ((col == 4) || (col == 5))  {
			pil = C;
		} else if ((col == 12) || (col == 13)) {
			pil = D;
		} else if ((col == 17) || (col == 16)) {
			pil = F;
		}
	} else if (col == 2) {
		if ((row >= 3) && (row <= 8)) {
			pil = B;
			subRow = row - 3;
		}
	}

	if (pil != -1) {
		for (int i = 0; i < 7; i++) {
			float c = i * (255.0f / 8.0f) + 10;
			addColor(pil, i, subRow, c, 0, 0);
			addColor(pil, i, subRow, 0, c, 0);
			addColor(pil, i, subRow, 0, 0, c);
			addColor(pil, i, subRow, c, c, c);
		}
	}

  
}