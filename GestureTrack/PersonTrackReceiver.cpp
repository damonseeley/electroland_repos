#include <limits.h>
#include "PersonTrackReceiver.h"

PersonTrackReceiver::PersonTrackReceiver(string moderatorIP){
	float PixelsPerUnit	= 1.0;
	float LeftCoord	= 0.0;
	float Width	= 0.0;
	float BottomCoord =	0.0;
	float Height = 0.0;
	Units = PersonTrackAPI::tyzxCentimeters;

	trax = newPersonTrackAPIFactory();
		trax->setWarningErrorFun(myWarningErrorFun);
	trax->setFatalErrorFun(&myFatalErrorFun);
	trax->initializeSettings();

	// read	values from	TyzxWorldCoords.ini	and	TrackWorldMap.ini, if any
	trax->ingestTrackWorldBounds(LeftCoord,	BottomCoord, Width,	Height,	Units, PixelsPerUnit);

	char* modIP = new char[moderatorIP.size() + 1];
	strcpy(modIP, moderatorIP.c_str());
	trax->setServerIP(modIP);

		trax->setAutoRestart(true);
	//trax->setUnits(Traxess::tyzxCentimeters);	// deprecated
	trax->setUnits(Units);
	bool enableTrackData = true;
	bool enableCameraDescs = true;
	bool enableObjectDescs = true;
	bool enableSourceImages	= true;

	if (enableTrackData)
		trax->enableTrackData();
	if (enableCameraDescs)
		trax->enableCameraDescs(true);
	if (enableObjectDescs)
		trax->enableObjectDescs();

	
}
void PersonTrackReceiver::start(){
	bool enableTrackData = true;
	bool enableCameraDescs = true;
	bool enableObjectDescs = true;
	bool enableSourceImages	= true;

	if (enableTrackData)
		trax->enableTrackData();
	if (enableCameraDescs)
		trax->enableCameraDescs(true);
	if (enableObjectDescs)
		trax->enableObjectDescs();

	while (!trax->probeConfiguration())
	{
		printf("No moderators responding, type 'y' to try again.\n");
		char buffer[512];
		fgets(buffer, 512, stdin);
		if (buffer[0] != 'y')
			exit(0);
		printf("Probing	'%s'.\n", trax->getServerIP());
	}

	trax->startCapture();



	printf("Started	capture	(at	most %d	cameras)\n", trax->getMaxCameras());

}
void PersonTrackReceiver::grab(TrackHash *hash){

		stat = trax->grab();
		hash->clear();

		int	i;

		if (trax->areCameraDescsEnabled())
		{
			int	nCameras;

			CameraDescBlock	*cdb = trax->getCameraDescBlock(nCameras);
			CameraDesc *cds	= cdb->cameraDescs;

			if (nCameras > 0)
			{
				fprintf(stdout,	"WorldBounds({{%5.3f, %5.3f}, {%5.3f, %5.3f}});\n",	
					cdb->worldBounds[0][0],	cdb->worldBounds[0][1],
					cdb->worldBounds[1][0],	cdb->worldBounds[1][1]);
			}

			for	(i = 0;	i <	nCameras; i++, cds++)
			{
				if (cds->online) {
					fprintf(stdout,	"Camera(%d,	%d,	%s,	%15.2lf,\n",
						cds->id, cds->online, cds->cameraName, cds->timeStamp);
					fprintf(stdout,	"\t%6.1f, %6.1f, %6.1f,	%5.3f, %5.3f,\n",
						cds->x,	cds->y,	cds->height, cds->thetaX, cds->thetaY);
					fprintf(stdout,	"\t%5.3f, %5.3f, %5.3f,	%5.3f, %d, %d,\n",
						cds->thetaZ, cds->cX, cds->cY, cds->cZ,	cds->imageWidth, cds->imageHeight);
					fprintf(stdout,	"\t{");
					int	i;
					for	(i = 0;	i <	4; i++)
						fprintf(stdout,	"{%6.1f, %6.1f},", cds->bounds[i][0], cds->bounds[i][1]);
					fprintf(stdout,	"});\n");
				} else { //	offline
					fprintf(stdout,	"Camera(%d,	%d,	%s,	%15.2lf);\n",
						cds->id, cds->online, cds->cameraName, cds->timeStamp);
				}
			}
		}

		ObjectDescType odType;
		if (trax->areObjectDescsEnabled(odType)) {
			int	nObjects;

			ObjectDesc *objectDescs	= trax->getObjectDescs(nObjects);


			for	(i = 0;	i <	nObjects; i++, objectDescs++) {
				if(objectDescs->objState ==	entryState)	{
					hash->addEnter(objectDescs->id);
				} else {
					hash->addExit(objectDescs->id);
				}
			}	
		}

		curFrame++;
		TrackPtType	tdType;
		if (trax->isTrackDataEnabled(tdType))
		{
			int	nPoints;
			TrackDataBlock *tdb	= trax->getTrackDataBlock(nPoints);
			for(i = 0; i < nPoints; i++, tdb++) {
			TrackPt	*tp	= tdb->trackPts;
			hash->updateTrack(tp->id, tp->x * scale, tp->y * scale, tp->h * scale, LONG_MAX );
			}

		}



}
PersonTrackReceiver::~PersonTrackReceiver() {
	delete trax;
}

bool PersonTrackReceiver::myWarningErrorFun(const char	*message)
{
	printf("Warning	Error: %s\n",message);
	return true; //	continue after warning
}

void PersonTrackReceiver::myFatalErrorFun(const char *message)
{
	printf("Fatal Error: %s\n",message);
}
