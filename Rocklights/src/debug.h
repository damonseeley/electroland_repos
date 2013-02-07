/*
 *  debug.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/28/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */


// file debug.h

#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <iostream>

#include "globals.h"

#ifdef _MYDEBUG

#define ASSERT(XX, YY) {if(!(XX)) { std::cout << "ASSERT FAILED: " << YY << std::endl; exit(-1) ; }}
#define DEBUGMSG() { std::cerr << std::endl << "---------------------------------" << std::endl << " WARNING: RUNNING DEBUG COMPILE" << std::endl << "---------------------------------" << std::endl; std::cin;} 
#else

#define ASSERT(XX, YY)
#define DEBUGMSG() 
#endif

#endif // __DEBUG_H__