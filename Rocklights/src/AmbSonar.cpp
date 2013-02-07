#include "ambSonar.h"
#include "SoundHash.h"

int AmbSonar::curCol = -1;

AmbSonar::AmbSonar(bool ci) : Ambient(ci)
{
	msPerStick = 5;
	timeLeftForStick = 0;
	width = 10;
	curStick = -21 ;
	lastPing = -1000;
}

AmbSonar::~AmbSonar(void)
{
	curCol = -1;
}


void AmbSonar::updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps)  {
	if(timeLeftForStick <= 0) {
		curStick++;
		if(curStick > ((Panels::thePanels->getWidth(Panels::A) + 7) * 4) + width) {
			curStick = -21 ;
			lastPing = -1000;
				;
		};

		timeLeftForStick = msPerStick + timeLeftForStick; // for overrun
	} else {
		timeLeftForStick -= dt;
	}

	for(int i = 0; i < width; i++) {
		int color = (255.0 * (width - i)) / ((float) width);

		int co = (curStick -i) / 4;
		int so = (curStick -i) % 4;

		if (co <= 7) {
			int c = 7 - co;
			int s = 3 - so;
			for(int r = 0; r < Panels::thePanels->getHeight(Panels::B); r++) {
				BasePixel *p = Panels::thePanels->getPixel(Panels::B, c, r);
				if(! p->isTarget) {
					p->addColor(s, 0, color, 0);
				}
			}
		} 
		if (co >= 6) {
			int c = co -=7;
			curCol = c;
			int s = so;

			for(int r = 0; r < Panels::thePanels->getHeight(Panels::A); r++) {
				BasePixel *p = Panels::thePanels->getPixel(Panels::A, c, r);

				SubPixel *sp = (SubPixel *) p->getSubPixel(s);
				if(sp->b < 0) {
					p->addColor(- sp->r + color, 0, 0);
					if((color == 255) && (lastPing <= curStick - 4)) {
						SoundHash::theSoundHash->play("ding.wav");
						lastPing = curStick;
					}
				} else if(! p->isTarget) {
					sp->addColor(0, color, 0);
				}
			}
		}
	}

}
