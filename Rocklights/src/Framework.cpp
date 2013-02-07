/*
 * Copyright (c) 2005, Creative Labs Inc.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided
 * that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and
 * 	     the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions
 * 	     and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *     * Neither the name of Creative Labs Inc. nor the names of its contributors may be used to endorse or
 * 	     promote products derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// Win32 version of the Creative Labs OpenAL 1.1 Framework for samples
#include<windows.h>
#include<stdio.h>

#include "Framework.h"

extern ALuint ALmain();

#define MAX_DEVICES 12

// Global variables;
static const ALchar *g_defaultDevice = NULL;
static const ALchar *g_deviceList = NULL;
static const ALchar *g_devices[MAX_DEVICES];
static ALCdevice *g_currentDevice = NULL;
static ALCcontext *g_currentContext = NULL;
static ALuint g_numDevices = 0;
static ALuint g_numDefaultDevice = 0;

ALvoid DisplayALError(ALchar *text, ALint errorcode)
{
    ALchar str[ 256 ];
	sprintf( str, "%s%s\n", text, alGetString(errorcode) );
    printf( str );
}

char output_str[256];
void ALprintf( const char* x, ... )
{
    va_list args;
    va_start( args, x );
    vsprintf( output_str, x, args ); 
    va_end( args );
	printf(output_str);
}

ALchar fullPath[256];
ALchar *ALaddPath(const ALchar *filename)
{
	sprintf(fullPath, "%s%s", "..\\..\\Media\\", filename);
	return fullPath;
}

ALboolean ALinit(ALchar *device)
{
	ALint lMajor, lMinor;

	//ALchar *aString;
	g_currentDevice = alcOpenDevice(device);
	if(NULL == g_currentDevice)
	{
		return AL_FALSE;
	}
	alcGetError( g_currentDevice );
	g_currentContext = alcCreateContext(g_currentDevice, NULL);
	if(NULL == g_currentContext)
	{
		return AL_FALSE;
	}
	alcMakeContextCurrent(g_currentContext);

	alcGetIntegerv( g_currentDevice, ALC_MAJOR_VERSION, 1, &lMajor );
	alcGetIntegerv( g_currentDevice, ALC_MINOR_VERSION, 1, &lMinor );
	printf( "\nOpen AL Version %d.%d\n", lMajor, lMinor );

	printf("Renderer : %s\n", (ALchar*)alGetString(AL_RENDERER));
	printf("Vendor : %s\n", (ALchar*)alGetString(AL_VENDOR));
	return AL_TRUE;
}

void ALclose()
{
    alcDestroyContext(g_currentContext);
    alcCloseDevice(g_currentDevice);
}

void main(void)
{
	// Print some info
	printf("Open AL sample framework\n");
	printf("------------------------\n\n");

	if(AL_FALSE == ALinit(NULL))
	{
		return;
	}

	alGetError();
	alcGetError( g_currentDevice );

	ALmain();

	ALclose();
}