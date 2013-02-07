#include "AvatarCreatorHash.h"
#include "AVSonarPinger.h"
#include "AVHuge.h"
#include "AV1SquareBUS.h"
#include "AVPlusSign.h"
#include "AVSinglePixelDance.h"
#include "AV1SquarePulse.h"

AvatarCreatorHash *AvatarCreatorHash::theACHash = NULL;
AvatarCreator *AvatarCreatorHash::defaultCreator = NULL;

AvatarCreatorHash::AvatarCreatorHash(void)
{
	if (theACHash == NULL) {
		theACHash = this;
		defaultCreator = new AC1Square();
		add(defaultCreator);
		add(new ACPlusSign());
		add(new AC9Square());
		add(new ACGeneric());
		add(new ACSonarPinger());
		add(new ACHuge());
		add(new ACSquareBUS());
		add(new ACSinglePixelDance());
		add(new AC1SquarePulse());
	}
}

AvatarCreatorHash::~AvatarCreatorHash(void)
{
}

void AvatarCreatorHash::add(AvatarCreator* ac) {
	AvatarCreator* dup = get(ac->getName());
	if(dup == NULL) {
		cout << "Avatar: " << ac->getName() << "\n";
		creators.insert(entry(ac->getName(), ac));
	} else {
		timeStamp(); clog << "WARNING  Attempt to add AvatarCreator wilth duplicate name " << dup->getName() << "\n";
	}
}

AvatarCreator* AvatarCreatorHash::get(string name) {
		hash_map<string,AvatarCreator*>::iterator it;
		it = creators.find(name);
		if(it != creators.end()) return it->second;
		return NULL; // if not found
}

Avatar* AvatarCreatorHash::create(string name, PersonStats *personStats,  Interpolators *interps) {
	AvatarCreator* ac = get(name);
	if (ac != NULL) return ac->create(personStats, interps);
	// if creater not registered
	timeStamp(); clog << "WARNING  AvatarCreator not found for name " << name << "using default avatar type\n";
	return defaultCreator->create(personStats, interps);
}