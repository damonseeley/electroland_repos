/**************************************************************************
 *
 *	TyzxTCPreceiver.cpp
 *
 *	Description	:	Demo code for TCP sending of images
 *
 *	Copyright (c) 2008-2009, Tyzx Corporation. All rights reserved.
 *
 **************************************************************************/

# include <stdio.h>
# include <stdlib.h>
# ifdef WIN32
# include <winsock2.h>
# else
# include <arpa/inet.h>
# include <memory.h>
# endif

# include "TyzxTCPreceiver.h"

# include "DeepSeaIF.h"

TyzxTCPreceiver::TyzxTCPreceiver(const char *servIP_, int servPort_)
{
	servIP = servIP_;
	servPort = servPort_;

	settingsInitialized = false;
	configurationProbed = false;
	captureStarted = false;

	int i;

	for (i = 0; i < MaxImages; i++) {
		imageBuffers[i] = NULL;
		imageLengthsInBytes[i] = 0;
	}
	
	compressedDataBuf = NULL;
	compressedDataBufLen = 0;

	eightBitImagesEnabled = true; // by default
	memset(grabbedImages, 0, NwhichImages * sizeof(void *));
	memset(widths, 0, NwhichImages * sizeof(int));
	memset(heights, 0, NwhichImages * sizeof(int));
}

TyzxTCPreceiver::~TyzxTCPreceiver()
{
	int i;

	for (i = 0; i < MaxImages; i++) {
		if (imageBuffers[i])
			delete [] imageBuffers[i];
		imageBuffers[i] = NULL;
		imageLengthsInBytes[i] = 0;
	}
	
	if (compressedDataBuf)
		delete [] compressedDataBuf;
	compressedDataBufLen = 0;


}

bool 
TyzxTCPreceiver::initializeSettings(char inifile[])
{
	if (settingsInitialized) {
		fprintf(stderr, "Cannot initialize settings multiple times.\n");
		return false;
	}

	settingsInitialized = true;

	return true;
}

bool 
TyzxTCPreceiver::probeConfiguration()
{
	if (!settingsInitialized) {
		fprintf(stderr, "You must initialize settings before probing configuration.\n");
		return false;
	}

	if (configurationProbed) {
		fprintf(stderr, "Cannot probe configuration multiple times.\n");
		return false;
	}
	
	int retVal  = connect(servIP, servPort);

	if (retVal < 0) {
		fprintf(stderr, "Connect to '%s' on port %d failed.\n", servIP, servPort);
		return false;
	}

	retVal = getParameters(&parameters);

	if (retVal < 0) {
		fprintf(stderr, "getParameters failed.\n");
		return false;
	}

	configurationProbed = true;

	return true;
}

bool 
TyzxTCPreceiver::startCapture()
{
	if (captureStarted) {
		fprintf(stderr, "You cannot call startCapture multiple times.\n");
		return false;
	}

	if (!settingsInitialized) {
		fprintf(stderr, "You must initialize settings before starting capture.\n");
		return false;
	}

	if (!configurationProbed) {
		bool result = probeConfiguration();
		if (!result)
			return false;
	}

	captureStarted = true;

	return true;
}

bool 
TyzxTCPreceiver::grab()
{
	int retVal = sendRequest(GRAB_NEXT_FRAME);

	if (retVal != 0)
		return recordError(retVal);

	retVal = getFrameInformation(&frameInformation);

	if (retVal != 0)
		return recordError(retVal);

	bool lastImage = false;
	TCPexampleWhichImage whichImage;
	TCPexampleImageType imageType;
	int width; 
	int height;
	void *image;
	int nextImageBuffer = 0;

	do {
		retVal = sendRequest(GET_ONE_IMAGE);

		if (retVal != 0)
			return recordError(retVal);

		retVal = getImage(nextImageBuffer, whichImage, imageType, width, height, image, lastImage);

		widths[whichImage] = width;
		heights[whichImage] = height;

		if (retVal == 0)
			nextImageBuffer++;
		else {
			fprintf(stderr, "grab: getImage failed.\n");
			return recordError(retVal);
		}

		grabbedImages[whichImage] = image;
		if ((whichImage == LeftImage) || (whichImage == RightImage))
			eightBitImagesEnabled = imageType == EightBitIntensity;
		else if (whichImage == ColorImage)
			eightBitImagesEnabled = imageType == YUV2image;

	} while (!lastImage);

	return true;
}

bool 
TyzxTCPreceiver::isZEnabled()
{ 
	return grabbedImages[RangeImage] != NULL;
}

bool 
TyzxTCPreceiver::isLeftEnabled()
{
	return grabbedImages[LeftImage] != NULL;
}

bool 
TyzxTCPreceiver::isLeftColorEnabled()
{
	return grabbedImages[ColorImage] != NULL;
}

bool 
TyzxTCPreceiver::isRightEnabled()
{
	return grabbedImages[RightImage] != NULL;
}

bool 
TyzxTCPreceiver::are8bitImagesEnabled()
{
	return eightBitImagesEnabled;
}

ColorMode
TyzxTCPreceiver::getColorMode()
{
	return YUV2color;
}

bool 
TyzxTCPreceiver::isProjectionEnabled()
{
	return grabbedImages[ProjectionImage] != NULL;
}

int 
TyzxTCPreceiver::zWidth()
{
	return widths[RangeImage];
}

int 
TyzxTCPreceiver::zHeight()
{
	return heights[RangeImage];
}

int 
TyzxTCPreceiver::intensityWidth()
{
	return widths[LeftImage] | widths[RightImage] | widths[ColorImage];
}

int 
TyzxTCPreceiver::intensityHeight()
{
	return heights[LeftImage] | heights[RightImage] | heights[ColorImage];
}

int 
TyzxTCPreceiver::projectionWidth()
{
	return widths[ProjectionImage];
}

int 
TyzxTCPreceiver::projectionHeight()
{
	return heights[ProjectionImage];
}

unsigned short *
TyzxTCPreceiver::getZImage()
{
	return (unsigned short *) grabbedImages[RangeImage];
}

unsigned char *
TyzxTCPreceiver::getLImage()
{
	return eightBitImagesEnabled ? (unsigned char *) grabbedImages[LeftImage] : NULL;
}

unsigned short *
TyzxTCPreceiver::getLImage16()
{
	return !eightBitImagesEnabled ? (unsigned short *) grabbedImages[LeftImage] : NULL;
}

unsigned char *
TyzxTCPreceiver::getColorImage()
{
	return eightBitImagesEnabled ? (unsigned char *) grabbedImages[ColorImage] : NULL;
}
	
unsigned short *
TyzxTCPreceiver::getColorImage16()
{
	return !eightBitImagesEnabled ? (unsigned short *) grabbedImages[ColorImage] : NULL;
}

unsigned char  *
TyzxTCPreceiver::getRImage()
{
	return eightBitImagesEnabled ? (unsigned char *) grabbedImages[RightImage] : NULL;
}

unsigned short *
TyzxTCPreceiver::getRImage16()
{
	return !eightBitImagesEnabled ? (unsigned short *) grabbedImages[RightImage] : NULL;
}

unsigned short *
TyzxTCPreceiver::getProjectionImage()
{
	return (unsigned short *) grabbedImages[ProjectionImage];
}

int 
TyzxTCPreceiver::getFrameNumber()
{
	return frameInformation.frameNumber;
}

double 
TyzxTCPreceiver::getFrameTime()
{
	return frameInformation.timeStamp;
}

int 
TyzxTCPreceiver::getZUnits()
{
	return (int) parameters.zUnits;
}

void 
TyzxTCPreceiver::getZimageCenter(double& u, double& v)
{
	u = parameters.zImageCenterU;
	v = parameters.zImageCenterV;
}

void 
TyzxTCPreceiver::getIntensityImageCenter(double& u, double& v)
{
	u = parameters.appearanceImageCenterU;
	v = parameters.appearanceImageCenterV;
}

double TyzxTCPreceiver::estimatedDisparityPrecisionPixels()
{
	return parameters.disparityPrecision;
}

int TyzxTCPreceiver::getMaxIntensityBits()
{
	return parameters.maxIntensityBits;
}

int TyzxTCPreceiver::getNCorrelators()
{
	return parameters.numCorrelators;
}

double TyzxTCPreceiver::getCz()
{
	return parameters.cZ;
}

double 
TyzxTCPreceiver::getCx()
{
	return parameters.cX;
}

double 
TyzxTCPreceiver::getCy()
{
	return parameters.cY;
}

int 
TyzxTCPreceiver::getParameters(TCPexampleParameters *parameters)
{
	int retVal = sendRequest(GET_PARAMETERS);

	if (retVal != 0)
		return retVal;

	retVal = getParameterBuffer(parameters);

	return retVal;
}

int 
TyzxTCPreceiver::sendRequest(TCPrequest request)
{
	return send(&request, 1);
}

int 
TyzxTCPreceiver::getHeader(TCPexampleHeader *header)
{
	int retVal = recv(header, sizeof(TCPexampleHeader));

# ifndef __BIG_ENDIAN__
	// need to convert to host order (LITTLE_ENDIAN)
	header->height = ntohs(header->height);
	header->width = ntohs(header->width);
	header->lengthInBytes = ntohl(header->lengthInBytes);
# endif

	return retVal;
}

int 
TyzxTCPreceiver::getParameterBuffer(TCPexampleParameters *parameters)
{
	TCPexampleHeader header;
	int retVal = getHeader(&header);
	if (retVal < 0)
		return retVal;
	retVal = recv(parameters, sizeof(TCPexampleParameters));

# ifndef __BIG_ENDIAN__
	// need to convert to host order (LITTLE_ENDIAN)
	parameters->cX = swizzleDoubleFun(parameters->cX);
	parameters->cY = swizzleDoubleFun(parameters->cY);
	parameters->cZ = swizzleDoubleFun(parameters->cZ);
	parameters->zUnits = swizzleDoubleFun(parameters->zUnits);
	parameters->zImageCenterU = swizzleDoubleFun(parameters->zImageCenterU);
	parameters->zImageCenterV = swizzleDoubleFun(parameters->zImageCenterV);
	parameters->appearanceImageCenterU = swizzleDoubleFun(parameters->appearanceImageCenterU);
	parameters->appearanceImageCenterV = swizzleDoubleFun(parameters->appearanceImageCenterV);
	parameters->disparityPrecision = swizzleDoubleFun(parameters->disparityPrecision);
	parameters->numCorrelators = ntohl(parameters->numCorrelators);
	parameters->maxIntensityBits = ntohl(parameters->maxIntensityBits);
# endif

	return retVal;
}

int 
TyzxTCPreceiver::getFrameInformation(TCPexampleFrameInformation *frameInformation)
{
	int retVal = recv(frameInformation, sizeof(TCPexampleFrameInformation));

# ifndef __BIG_ENDIAN__
	// need to convert to host order (LITTLE_ENDIAN)
	frameInformation->frameNumber = ntohl(frameInformation->frameNumber);
	frameInformation->timeStamp = swizzleDoubleFun(frameInformation->timeStamp);
# endif
	return retVal;
}

int 
TyzxTCPreceiver::getImage(int imageIndex, 
				  TCPexampleWhichImage &whichImage, 
				  TCPexampleImageType &imageType,
				  int &width, int &height, void *&image,
				  bool &lastImage)
{
	if ((imageIndex < 0) || (imageIndex >= MaxImages)) {
		printf("ImageIndex out of bounds %d\n", imageIndex);
		return -1;
	}

	TCPexampleHeader header;

	int retVal = getHeader(&header);

	if (retVal < 0)
		return retVal;

	whichImage = (TCPexampleWhichImage) header.whichImage;
	imageType = (TCPexampleImageType) header.imageType;
	width = header.width;
	height = header.height;
	lastImage = !!(header.imageProperties & LastImage);

	if (header.lengthInBytes <= 0)
		return retVal;

	bool compressed = !!(header.imageProperties & Compressed);
	bool littleEndian = !!(header.imageProperties & LittleEndian);
	void *bytesToReceive;

	unsigned long fullLength = fullLength = header.width * header.height * header.bytesPerPixel;

	if (imageLengthsInBytes[imageIndex] < fullLength) {
		if (imageBuffers[imageIndex]) {
			delete [] imageBuffers[imageIndex];
		}
		imageLengthsInBytes[imageIndex] = fullLength;

		imageBuffers[imageIndex] = new unsigned char[fullLength + 16]; // padding for decompression
	}

	if  (compressed) {
		if (compressedDataBufLen < header.lengthInBytes) {
			if (compressedDataBuf) {
				delete [] compressedDataBuf;
				compressedDataBufLen = 2 * compressedDataBufLen;
			} else {
				compressedDataBufLen = header.lengthInBytes * 2;
			}
			compressedDataBuf = new unsigned long[compressedDataBufLen / 4];
		}
		bytesToReceive = compressedDataBuf;
	} else {
		bytesToReceive = imageBuffers[imageIndex];
	}

	retVal = recv(bytesToReceive, header.lengthInBytes);

	if (retVal == 0) {
		if (compressed) {
		  int height = header.height;
		  int width = header.width;
		  int bytesPerPixel = header.bytesPerPixel;
		  if ((header.imageType == YUV2image) || (header.imageType == SixteenBitYUV2image)) {
			  width *= 2;
			  bytesPerPixel /= 2;
		  }
		  DeepSeaIF::decompressImage((tyzxU32 *) bytesToReceive, 
			header.lengthInBytes / sizeof(tyzxU32), 
			width, height, bytesPerPixel, 
			imageBuffers[imageIndex]); 
		} else {
# ifdef __BIG_ENDIAN__
			if (littleEndian && (header.bytesPerPixel == 2)) {
# else
			if (!littleEndian && (header.bytesPerPixel == 2)) {
# endif
			    // swap every other byte.
			    char *data = (char *) imageBuffers[imageIndex];
			    int i;
			    int len = header.width * header.height;
			    for (i = 0; i < len; i++) {
			      char tmp = data[0];
			      data[0] = data[1];
			      data[1] = tmp;
			      data += 2;
			    }
			}
		}	
		image = imageBuffers[imageIndex];
	}
	return retVal;
}
