#ifndef _FRAMEWORK_H_ // Win32 version
#define _FRAMEWORK_H_

// Get some classic includes
#include<Windows.h>
#include<stdio.h>

#include"al.h"
#include"alc.h"
#include"alut.h"

extern ALvoid DisplayALError(ALchar *text, ALint errorcode);
extern ALvoid ALprintf( const ALchar * x, ... );
extern ALchar *ALaddPath(const ALchar *filename);

#endif _FRAMEWORK_H_