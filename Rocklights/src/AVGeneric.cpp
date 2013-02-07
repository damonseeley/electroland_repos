#include "AVGeneric.h"


AVGeneric::AVGeneric(PersonStats *personStats, Interpolators *interps) : Avatar () {
}


void AVGeneric::init(Interpolators *interps) {

	if(overheadArrangement != NULL) {
		overheadArrangement->apply(this, interps);
	}


	if((enterSound != "") && (enterSoundLoop <= 0)) {
				loopNumber = SoundHash::theSoundHash->play(enterSound, enterSoundLoop);
	} else {
		loopNumber = -1;
	}
}
void AVGeneric:: enter(Interpolators *interps) {
		if(enterArrangement != NULL) {
			enterArrangement->apply(this, interps);
		}
		if((enterSound != "") && (enterSoundLoop > 0)) {
			loopNumber = SoundHash::theSoundHash->play(enterSound, enterSoundLoop);
		} 
}



void AVGeneric::updateFrame(PersonStats *personStats, int ct, int dt, Interpolators *interps) {
	if(unEntered) {
		enter(interps);
		unEntered = false;
	}
	if(overheadArrangement != NULL) {
		overheadArrangement->updateFrame(this, personStats, ct, dt, interps);
	}


}

void AVGeneric::move(int col, int row, Interpolators *interps) {
	if(moveArrangement != NULL) {
		moveArrangement->apply(this, interps);
	}
	if(moveSound != "") {
		SoundHash::theSoundHash->play(moveSound);
	}
}
void AVGeneric::exit(Interpolators *interps) {
//	if(exitArrangement != NULL) {
//		exitArrangement->exit(getCol(), getRow(), interps);
//	}
	if(loopNumber >= 0) {
		SoundHash::theSoundHash->stop(loopNumber);
	}

	if((! unEntered) &&(exitSound != "")) {
		SoundHash::theSoundHash->play(exitSound);
	}
}


AVGeneric::~AVGeneric() {

}