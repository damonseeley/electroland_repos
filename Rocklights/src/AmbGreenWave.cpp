#include "ambgreenwave.h"
#include "MasterController.h"

int AmbGreenWave::curWaveCnt = 0;
int AmbGreenWave::curCols[MAXWAVES] = { -1,-1,-1,-1, -1};

AmbGreenWave::AmbGreenWave(bool ci) : Ambient(ci)
{
	curWave = curWaveCnt++;
//	cout << "====> created:curWave = " << curWaveCnt << endl;
	colString = "";
	resetParams();
}

AmbGreenWave::~AmbGreenWave(void)
{
	curCols[curWave] = -1;
	curWaveCnt--;
//	cout << "====> deleted:curWave = " << curWaveCnt << endl;
}

void AmbGreenWave::resetParams() {
	string setting = MasterController::curMasterController->name + colString + "WaveMsPerStick";
	msPerStick = CProfile::theProfile->Int(setting.c_str(), 1);
	timeLeftForStick = 0;
	setting = MasterController::curMasterController->name + colString +"WaveWidth";
	width = CProfile::theProfile->Int(setting.c_str(), 6);
	setting = MasterController::curMasterController->name + colString +"WaveStepSize";
	stepSize = CProfile::theProfile->Int(setting.c_str(), 1);
//	cout << setting << " = " <<stepSize << endl;;
	maxSize = ((Panels::thePanels->getWidth(Panels::A) + 14) * 4) + width;
	curStick = random(maxSize);
}
// pan c j = col 5
// pan d i= col 12
// e,f g, h are end

void AmbGreenWave::updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps)  {
	if(timeLeftForStick <= 0) {
		curStick += stepSize;
		if(curStick > maxSize) {
			curStick = 0 ;
			;
		};

		timeLeftForStick = msPerStick + timeLeftForStick; // for overrun
	} else {
		timeLeftForStick -= dt;
	}
	for(int i = 0; i < width; i++) {
		int color = (255.0 * (width - i)) / ((float) width);
		int cr = r * color;
		int cg = g * color;
		int cb = b * color;

		int co = curStick-i;
		if(co >= 0) co = (curStick -i) / 4;
		int so = (curStick -i) % 4;

		if ((co >= 0) && (co < 7)) {
			int c = 6 - co;
			int s = 3 - so;
			for(int r = 0; r < Panels::thePanels->getHeight(Panels::B); r++) {
				BasePixel *p = Panels::thePanels->getPixel(Panels::B, c, r);
				if(! p->isTarget) {
					p->addColor(s, cr, cg, cb);
				}
			}
		} 
		if (co > 4) {
			int c = co -5;
			curCols[curWave] = c;
			int s = so;
			if(c <= 19) {

				if((c == 5) && (s == 2)) {
					for(int pilC = 0; pilC < 7; pilC++) {
						BasePixel *p = Panels::thePanels->getPixel(Panels::C, pilC, 0);
						if(! p->isTarget) p->addColor(cr,cg, cb);
						p = Panels::thePanels->getPixel(Panels::J, pilC, 0);
						if(! p->isTarget) p->addColor(cr,cg, cb);
					}
				} else if((c == 12) && (s == 2)) {
					for(int pilC = 0; pilC < 7; pilC++) {
						BasePixel *p = Panels::thePanels->getPixel(Panels::D, pilC, 0);
						if(! p->isTarget) p->addColor(cr,cg, cb);
						p = Panels::thePanels->getPixel(Panels::I, pilC, 0);
						if(! p->isTarget) p->addColor(cr,cg, cb);
					}
				}
				for(int r = 0; r < Panels::thePanels->getHeight(Panels::A); r++) {
					BasePixel *p = Panels::thePanels->getPixel(Panels::A, c, r);
					if(! p->isTarget) {
						p->addColor(s, cr, cg, cb);
					}
				}
			} 
			if(c >= 18) {
				c = c-19;
				if((c >=0) && (c < 8)) {
						BasePixel *p = Panels::thePanels->getPixel(Panels::E, c, 0);
						if(! p->isTarget) p->addColor(s,cr,cg, cb);
						p = Panels::thePanels->getPixel(Panels::F, c, 0);
						if(! p->isTarget) p->addColor(s,cr,cg, cb);
						p = Panels::thePanels->getPixel(Panels::G, c, 0);
						if(! p->isTarget) p->addColor(s,cr,cg, cb);
						p = Panels::thePanels->getPixel(Panels::H, c, 0);
						if(! p->isTarget) p->addColor(s,cr,cg, cb);
				}


			}
		}
	}

}
