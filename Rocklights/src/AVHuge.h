/*
 *  AVPerson.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/30/05.
 * 
 *
 */
#ifndef __AVSMALL_H__
#define __AVSMALL_H__

#include "Avatar.h"
#include "PersonStats.h"
#include "Panels.h"
#include "Interpolators.h"
#include "IGeneric.h"
#include "BasePixel.h"
#include "AmbientA.h"
#include "AvatarCreator.h"

class Avatar; 

class AVHuge : public Avatar  {
  
// static int seasR[];
 static int seas[];
 static int seasLight[];
 bool isGreen;

 static bool init;
 public:
  AVHuge(PersonStats *personStats,  Interpolators *interps);
  void updateFrame(PersonStats *personStats,  int ct, int dt, Interpolators *interps);
}
;
class ACHuge : public AvatarCreator {
public:
	ACHuge(void) {  };
	~ACHuge(void) {};
	
	virtual Avatar* create(PersonStats *personStats,  Interpolators *interps) {
		return new AVHuge(personStats, interps);
	}

	virtual string getName() { return "AvHuge"; };

};
#endif