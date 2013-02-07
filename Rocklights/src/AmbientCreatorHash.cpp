#include "AmbientCreatorHash.h"
#include "AmbientA.h"
#include "AmbGreenWave.h"
#include "AmbSonar.h"
#include "AmbWillyWonka.h"
#include "AmbTron.h"
#include "AmbKit.h"
#include "AmbSin.h"
#include "AmbComposer.h"
#include "AmbWashBang.h"
#include "AmbBGFlasher.h"
#include "AmbTargetFlash.h"
#include "AmbColorWash.h"
#include "AmbBlinkTag.h"
#include "AmbSine.h"
#include "AmbSound.h"

AmbientCreatorHash *AmbientCreatorHash::theACHash = NULL;
AmbientCreator *AmbientCreatorHash::defaultCreator = NULL;

AmbientCreatorHash::AmbientCreatorHash(void)
{
	if (theACHash == NULL) {
		theACHash = this;
		defaultCreator = new AmbCAmbEmpty();
		add(defaultCreator);
		add(new AmbCRedStick());
		add(new AmbCAmbientA());
		add(new AmbCGreenWave());
		add(new AmbCBlueWave());
		add(new AmbCRedWave());
		add(new AmbCSonar());
		add(new AmbCWillyWonka());
		add(new AmbCTron());
		add(new AmbCKitRed());
		add(new AmbCKitRedCrazy());
		add(new AmbCKitBlue());
		add(new AmbCKitBlueCrazy());
		add(new AmbCSin());
		add(new AmbCComposer());
		add(new AmbCWashBang());
		add(new AmbCBGFlasher());
		add(new AmbCTargetFlash());
		add(new AmbCColorWash());
		add(new AmbCBlinkTag());
		add(new AmbCSine());
		add(new AmbCSound());

	}
}

AmbientCreatorHash::~AmbientCreatorHash(void)
{
}

void AmbientCreatorHash::add(AmbientCreator* ac) {
	AmbientCreator* dup = get(ac->getName());
	if(dup == NULL) {
		cout << "Ambient: " << ac->getName() << "\n";
		creators.insert(entry(ac->getName(), ac));
	} else {
		timeStamp(); clog << "WARNING  Attempt to add AmbientCreator wilth duplicate name " << dup->getName() << "\n";
	}
}

AmbientCreator* AmbientCreatorHash::get(string name) {
		hash_map<string,AmbientCreator*>::iterator it;
		it = creators.find(name);
		if(it != creators.end()) return it->second;
		return NULL; // if not found
}

Ambient* AmbientCreatorHash::create(string name, bool createInterps) {
	AmbientCreator* ac = get(name);
	if (ac != NULL) return ac->create(createInterps);
	// if creater not registered
	timeStamp(); clog << "WARNING  AmbientCreator not found for name " << name << "using default Ambient type\n";
	return defaultCreator->create(createInterps);
}