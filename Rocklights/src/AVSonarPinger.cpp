#include "AVSonarPinger.h"
#include "AmbGreenWave.h"
#include "SoundHash.h"

int AVSonarPinger::red[] = {
	255,	0,	0,	0,		0,	0,	2800,
	-1};

AVSonarPinger::AVSonarPinger() : Avatar() {
	for(int i = 0; i < MAXWAVES; i++) {
		justPinged[i] = false;
	}
}
void AVSonarPinger::updateFrame(PersonStats *personStats, int ct, int dt, Interpolators *interps) {
	int col = getCol();
	for(int wave = 0; wave < AmbGreenWave::curWaveCnt; wave++) { 
	if(justPinged[wave]) {
		if(col != AmbGreenWave::curCols[wave]) {
			justPinged[wave] = false;
		}
	} else {
		if(col == AmbGreenWave::curCols[wave]) {
			new IGeneric(interps, Panels::thePanels->getPixel(Panels::A, col, getRow()), red);
			SoundHash::theSoundHash->play("ping.wav");
			justPinged[wave] = true;
		}
	}
	}	

}

