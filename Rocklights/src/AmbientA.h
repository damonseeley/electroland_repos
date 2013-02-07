/*
 *  Avatar .h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/30/05.
 * 
 *
 */

#ifndef __AMBIENTA_H__
#define __AMBIENTA_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"

class AmbientA  : public Ambient {
public:
  static int seas[];
  static int seasR[];
  static bool inited ;

  AmbientA(bool ci, bool green) ;
  virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) {}


} 
;

class AmbCAmbientA : public AmbientCreator {
public:
	AmbCAmbientA(void) {};
	~AmbCAmbientA(void) {};
	
	virtual Ambient* create(bool cI) {
		return new AmbientA(cI, true);
	}
	virtual string getName() { return "AmbBlueGreenSea"; };

};
#endif