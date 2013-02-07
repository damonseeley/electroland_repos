/*
 *  Interpolator.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */

#include "Interpolator.h"

Interpolator::Interpolator() {
  interpDoneListener = NULL;
  isGeneric = false;
}

Interpolator::~Interpolator() {

  if (interpDoneListener != NULL) {
    interpDoneListener->genericInterpDone(this);
    interpDoneListener = NULL;

  }
  /*  a regular interpolator is never instantiated
  and this is screwing up IGenerics destructor
  if(queue) {
    queue->remove(this); 
    queue = NULL;
  }
  */

}



