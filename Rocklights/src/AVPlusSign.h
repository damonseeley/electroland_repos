#ifndef __AVPLUSSIGN_H__
#define __AVPLUSSIGN_H__

#include "Avatar.h"
#include "IGeneric.h"
#include "AvatarCreator.h"

class Avatar; 

class AVPlusSign : public Avatar  {
  static int red[];
  static int green[];
  static int blue[];
  
public:
  AVPlusSign(PersonStats *personStats,  Interpolators *interps);
  void updateFrame(PersonStats *personStats,  int ct, int dt, Interpolators *interps);

}
;
class ACPlusSign : public AvatarCreator {
public:
	ACPlusSign(void) { 	 };
	~ACPlusSign(void) {};
	
	virtual Avatar* create(PersonStats *personStats,  Interpolators *interps) {
		return new AVPlusSign(personStats, interps);
	}

	virtual string getName() { return "PlusSign"; };

};


#endif