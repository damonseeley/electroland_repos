#include "ambsine.h"
#include "Panels.h"
#include "BasePixel.h"

int AmbSine::wave[] = {1,2,4,6,9,10,9,6,4,2,1,0};

AmbSine::AmbSine(bool ci) : Ambient(ci) { phase =0;phase2 =0; updateSpeed2 = 100; updateSpeed = 75; updateTime = 0; updateTime2 = 0;}
void AmbSine::updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) {
	
		BasePixel *p = Panels::thePanels->getPixel(Panels::A, 0, 0);
//			BasePixel *p = Panels::thePanels[Panels::A].getPixel(0,0);
			p->addColor(255,0,0);
			if(updateTime <=0) {
				phase++;
				updateTime = updateSpeed;
			} else {
				updateTime-=dt;
			}	
			if(updateTime2 <=0) {
				phase2++;
				updateTime2 = updateSpeed2;
			} else {
				updateTime2-=dt;
			}
	int x;
	for(int i = 0; i < 12; i++) {
		x = (i + phase) %13;
		int y = wave[x];
		y*=2;
		y+=8;
		for(int s = 0; s < y; s++) {
		BasePixel *p = Panels::thePanels->getPixel(Panels::A, s/4, i);
			p->addColor(s%4,0,255,0);
		}

	}
	
	for(int i = 0; i < 12; i++) {
		x = (i + phase2) %13;
		int y = wave[x];
		y*=4;
		y+=2;
		for(int s = 0; s < y; s++) {
		BasePixel *p = Panels::thePanels->getPixel(Panels::A, s/4, i);
			p->addColor(s%4,255,0,0);
		}

	}
}
