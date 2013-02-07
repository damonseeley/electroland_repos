

#ifndef __AMBKIT_H__
#define __AMBKIT_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"




class AmbKit : public Ambient {
public:
	
	static int kit1[];
	static int kit2[];
	static int blueKit1[];
	static int blueKit2[];

	AmbKit(bool ci) : Ambient(ci) {};
  void setup(bool isRed, bool lockStep);


} 
;
class AmbCKitRed : public AmbientCreator {
public:
	AmbCKitRed(void) {};
	~AmbCKitRed(void) {};
	
	virtual Ambient* create(bool ci) {
		AmbKit *a = new AmbKit(ci);
		a->setup(true, true);
		return a;
	}
	virtual string getName() { return "KitRed"; };

};
class AmbCKitBlue : public AmbientCreator {
public:
	AmbCKitBlue(void) {};
	~AmbCKitBlue(void) {};
	
	virtual Ambient* create(bool ci) {
		AmbKit *a = new AmbKit(ci);
		a->setup(false, true);
		return a;
	}
	virtual string getName() { return "KitBlue"; };

};
class AmbCKitRedCrazy : public AmbientCreator {
public:
	AmbCKitRedCrazy(void) {};
	~AmbCKitRedCrazy(void) {};
	
	virtual Ambient* create(bool ci) {
		AmbKit *a = new AmbKit(ci);
		a->setup(true, false);
		return a;
	}
	virtual string getName() { return "KitRedCrazy"; };

};
class AmbCKitBlueCrazy : public AmbientCreator {
public:
	AmbCKitBlueCrazy(void) {};
	~AmbCKitBlueCrazy(void) {};
	
	virtual Ambient* create(bool ci) {
		AmbKit *a = new AmbKit(ci);
		a->setup(false, false);
		return a;
	}
	virtual string getName() { return "KitBlueCrazy"; };

};

#endif