/**************************************************************************
 *
 *	TyzxTCPreceiver.h
 *
 *	Description	:	Class definitions for demo code for TCP sending of images
 *
 *	Copyright (c) 2008-2009, Tyzx Corporation. All rights reserved.
 *
 **************************************************************************/

# ifndef TYZX_TCP_RECEIVER_H_
# define TYZX_TCP_RECEIVER_H_

# include "TCPreceiver.h"
# include "TCPexample.h"

const int MaxImages = 10;

class TyzxTCPreceiver : public TCPreceiver
{
public:
	TyzxTCPreceiver(const char *servIP = "localhost", int servPort = TCP_EXAMPLE_PORT);
	~TyzxTCPreceiver();
	bool initializeSettings(char inifile[] = NULL);
	bool probeConfiguration();
	bool startCapture();
	bool grab();

	bool isZEnabled();
	bool isLeftEnabled();
	bool isLeftColorEnabled();
	bool isRightEnabled();
	bool are8bitImagesEnabled();
	ColorMode getColorMode();
	bool isProjectionEnabled();

	unsigned short *getZImage();
	unsigned char *getLImage();
	unsigned short *getLImage16();
	unsigned char *getColorImage();
	unsigned short *getColorImage16();
	unsigned char  *getRImage();
	unsigned short *getRImage16();
	unsigned short *getProjectionImage();

	int zWidth();
	int zHeight();
	int intensityWidth();
	int intensityHeight();
	int projectionWidth();
	int projectionHeight();

	int    getZUnits();
	void   getZimageCenter(double& u, double& v);
	void   getIntensityImageCenter(double& u, double& v);
	double estimatedDisparityPrecisionPixels();
	int    getMaxIntensityBits();
	int getNCorrelators();
	double getCx();
	double getCy();
	double getCz();
	
	int getFrameNumber();
	double getFrameTime();
	
	int getLastError() { return lastError; }
protected:
	int getParameters(TCPexampleParameters *parameters);
	int getHeader(TCPexampleHeader *header);
	int sendRequest(TCPrequest request);
	int getParameterBuffer(TCPexampleParameters *parameters);
	int getFrameInformation(TCPexampleFrameInformation *frameInformation);
	int getImage(int imageIndex, TCPexampleWhichImage &whichImage, 
				  TCPexampleImageType &imageType,
				  int &width, int &height, void *&image,
				  bool &lastImage);
	int lastError;
	bool recordError(int err) { lastError = err; return !err; }
	unsigned char *imageBuffers[MaxImages];
	unsigned long imageLengthsInBytes[MaxImages];
	unsigned long *compressedDataBuf;
	unsigned long compressedDataBufLen;
	TCPexampleParameters parameters;
	TCPexampleFrameInformation frameInformation;
	int servPort;
	const char *servIP;
	bool settingsInitialized;
	bool configurationProbed;
	bool captureStarted;
	bool eightBitImagesEnabled;
	void *grabbedImages[NwhichImages];
	int widths[NwhichImages];
	int heights[NwhichImages];
};
# endif
