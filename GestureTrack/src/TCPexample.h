/**************************************************************************
 *
 *	TCPexample.h
 *
 *	Description	:	Data structures and constants for TCP sending of images
 *
 *	Copyright (c) 2008-2009, Tyzx Corporation. All rights reserved.
 *
 **************************************************************************/

# include "DSTypes.h"

# ifndef TCP_EXAMPLE_H_
# define TCP_EXAMPLE_H_

# define TCP_EXAMPLE_PORT		6543 // arbitrary choice

// Protocol:
//  Sender (DSIF program) starts up DSIF in whatever modes the user desires, enabling
//		or disabling grabbing of Z, etc.
//  Sender opens server socket on specified port, and waits for a connection.
//  Receiver (user program on other end of net) connects to the Sender ip address and 
//		port
//  Sender waits to receive a request packet.
//	Receiver sends a request packet.  
//		Sender responds to a GET_PARAMETERS request by sending back a header tagged as MetaData,
//			followed by a TCPexampleParameters structure.  Goes back to waiting for a request packet.
//		Sender responds to a GRAB_NEXT_FRAME request by sending back a header tagged as MetaData,
//			followed by a TCPexampleFrameInformation structure.  
//		Sender waits for a GET_ONE_IMAGE command.  When a GET_ONE_IMAGE command is received,
//			Sender responds by sending a TCPexampleHeader.  If all enabled images have been returned
//				since the last GRAB_NEXT_FRAME call, TCPexampleHeader is tagged of type NoImage.
//			If there are still enabled images to send the TCPexampleFrameInformation is filled in
//				to describe the following image.  Following the header, the described image is sent. 
//              If this is the last image for this grab, the header
//				is also tagged with LastImage, and control goes back to the main grabbing loop.

enum TCPrequestType { GET_PARAMETERS = 1, GRAB_NEXT_FRAME = 2, GET_ONE_IMAGE = 3 };

typedef unsigned char TCPrequest;

enum TCPexampleWhichImage
{
	NoImage = 1,
	MetaData = 2,
	RangeImage = 3,
	LeftImage = 4,
	RightImage = 5,
	ProjectionImage = 6,
	ColorImage = 7,
	NwhichImages = 8 // keep up to count of these
};

enum TCPexampleImageType
{
	NotAnImageType = 0,
	SixteenBitRange = 1,
	EightBitIntensity = 2,
	SixteenBitIntensity = 3,
	SixteenBitProjectionCount = 4,
	Bytes = 5,
	YUV2image = 6,
	SixteenBitYUV2image = 7,
	Ntypes = 8,
};

enum TCPexampleImageProperties
{
	// bit fields
	Compressed = 0x1,
	LittleEndian = 0x2,
	LastImage = 0x4,
};

typedef struct
{
	unsigned char whichImage;
	unsigned char imageType;
	unsigned char imageProperties;
	unsigned char bytesPerPixel;
	unsigned short width;
	unsigned short height;
	tyzxU32  lengthInBytes; // if compressed, compressed length, otherwise product of bytesPerPixel * width * height
	// add more data here, make sure that data is aligned on 8 byte boundaries (some OSs use 4 bytes, some 8)
	// also make sure endian order is considered in getHeader in TyzxTCPreceiver.cpp
} TCPexampleHeader;

typedef struct
{
	double timeStamp;
	tyzxU32 frameNumber;
	tyzxU32 padding;
} TCPexampleFrameInformation;

typedef struct
{
	double cX;
	double cY;
	double cZ;
	double zUnits;
	double zImageCenterU;
	double zImageCenterV;
	double appearanceImageCenterU;
	double appearanceImageCenterV;
	double disparityPrecision;
        tyzxU32 maxIntensityBits; 
	tyzxU32 numCorrelators;
	// add more data here, make sure that data is aligned on 8 byte boundaries (some OSs use 4 bytes, some 8)
	// also make sure endian order is considered in getParameterBuffer in TyzxTCPreceiver.cpp
} TCPexampleParameters;

# ifdef __BIG_ENDIAN__
# define ntohd(d) d
# define htond(d) d
# else
inline double 
swizzleDoubleFun(double d)
{
	double temp;
	char *source = (char *) &d;
	char *dest = (char *) &temp;

	dest[0] = source[7];
	dest[1] = source[6];
	dest[2] = source[5];
	dest[3] = source[4];
	dest[4] = source[3];
	dest[5] = source[2];
	dest[6] = source[1];
	dest[7] = source[0];

	return temp;
}

# define ntohd(d) swizzleDoubleFun(d)
# define htond(d) swizzleDoubleFun(d)
# endif

# endif

