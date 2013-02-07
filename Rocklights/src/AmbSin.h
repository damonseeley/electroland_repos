

#ifndef __AMBSIN_H__
#define __AMBSIN_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"




class AmbSin : public Ambient {
public:
	static int wave[];


  AmbSin(bool ci) ;

} 
;
class AmbCSin : public AmbientCreator {
public:
	AmbCSin(void) {};
	~AmbCSin(void) {};
	
	virtual Ambient* create(bool ci) {
		return new AmbSin(ci);
	}
	virtual string getName() { return "Sin"; };

};
#endif