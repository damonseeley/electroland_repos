#include "Tracker.h"

// if there are people in the room AND no updates after a second report error

Tracker::Tracker(PeopleStats *ps, WorldStats *ws) {
  peopleStats = ps;
  worldStats = ws;
  trax = newPersonTrackAPIFactory();
  cameraCnt = 0;
  hasNewData = false;
  isModeratorDown = false;
  reportModeratorErrorTime = INT_MAX;
  reportModeratorEmptyRoomErrorTime = INT_MAX;
  reportModeratorEmptyRoomErrorInterval = CProfile::theProfile->Int("reportModeratorEmptyRoomErrorInterval", 2) * 60 * 60 * 1000;

  useOldCoordCalibration = CProfile::theProfile->Bool("useOldCoordCalibration", false);
  scaleX = CProfile::theProfile->Float("scaleX", 1.0f);
  scaleY = CProfile::theProfile->Float("scaleY", 1.0f);
  offsetX = CProfile::theProfile->Float("offsetX", 0.0f);
  offsetY = CProfile::theProfile->Float("offsetY", 0.0f);

  cutoffX = CProfile::theProfile->Float("cutoffX", FLT_MAX);

  excludeBoxE = CProfile::theProfile->Float("excludeBoxE", -4000);
  excludeBoxN = CProfile::theProfile->Float("excludeBoxN", FLT_MAX);
  excludeBoxS = CProfile::theProfile->Float("excludeBoxS", -4000);

  bounds = new Bounds();

  timeStamp(); std::clog << "INFO  Trax created\n";
}

bool Tracker::init(char *modIP) {

	//UNDONE - UNCOMMENT THIS CODE!
//	timeStamp(); std::cout << "ERROR  Trax error messages were commented out in Tracker.cpp uncomment an recompile" << endl;
//	timeStamp(); std::clog << "ERROR: Trax error messages were commented out in Tracker.cpp uncomment an recompile" << endl;
//			Globals::hasError = true;

  trax->setWarningErrorFun(errorMsg);
	trax->setFatalErrorFun(fatalMsg);

  trax->initializeSettings();
  trax->setServerIP(modIP);


		trax->enableTrackData();
		trax->enableCameraDescs(true);
		trax->enableObjectDescs();


    int attemps = TYZXATTACHATTEMPTS;

    while ((!trax->probeConfiguration()) && (attemps > 0)) {
      attemps--;
  		timeStamp(); std::clog << "WARNING  No moderators responding, retrying (attemp " << TYZXATTACHATTEMPTS - attemps << " of " << TYZXATTACHATTEMPTS << ")\n";
  		timeStamp(); std::clog << "WARNING  Probing " << trax->getServerIP() << "\n";
  	}
    if (attemps <= 0) {
     timeStamp(); std::clog << "ERROR  Unable to attach to monitor.  Not using tracking" << endl;
	 		Globals::hasError = true;

      return false;
    }
    
    trax->startCapture();
    
	timeStamp(); std::clog << "INFO  Started capture (at most " << trax->getMaxCameras() << " cameras\n";
	timeStamp(); std::cout << "INFO  Started capture (at most " << trax->getMaxCameras() << " cameras\n";
	timeStamp(); std::clog << "INFO  Trax is inited\n";
	timeStamp(); std::cout << "INFO  Trax is inited\n";
    

    return true;


}

// returns 1 if now data
// 0 if no new data
// -1 if tracking problem
bool Tracker::grab(int sleepTime) {
  
  int ret = trax->grab(sleepTime);
  if (ret > 0) {
    hasNewData = true;
	return true;
  } else if (ret < 0) {
    timeStamp(); clog << "WARNING: ReturnValue  " << ret << " encountered while attempting grab.\n";
	clearTrackingData();
	Globals::hasError = true;

  }
  return false;
}

void Tracker::checkCameras() {

  if (trax->areCameraDescsEnabled())
		{
			int nCameras;

			CameraDescBlock *cdb = trax->getCameraDescBlock(nCameras);
	
      if (nCameras > 0) {
  			CameraDesc *cds = cdb->cameraDescs;
		
			  for (int i = 0; i < nCameras; i++, cds++)
			  {
				  if (cds->online) {
					  cameraCnt++;
            timeStamp(); clog << "INFO  Camera " << cds->id << " online: " << cds->online << cds->cameraName << ", " << cds->timeStamp << "\n";
                         cout << "INFO  Camera " << cds->id << " online: " << cds->online << cds->cameraName << ", " << cds->timeStamp << "\n";
            timeStamp(); clog << "INFO         " << cds->x << ", " << cds->y << ", " << cds->height << ", " << cds->thetaX << ", " << cds->thetaY << "\n";
                         cout << "INFO         " << cds->x << ", " << cds->y << ", " << cds->height << ", " << cds->thetaX << ", " << cds->thetaY << "\n";
						 clog << "       " << cds->thetaZ << ", " << cds->cX << ", " << cds->cY << ", " << cds->cZ << ", " << cds->imageWidth << ", " << cds->imageHeight << "\n";
						 cout << "       " << cds->thetaZ << ", " << cds->cX << ", " << cds->cY << ", " << cds->cZ << ", " << cds->imageWidth << ", " << cds->imageHeight << "\n";
            timeStamp(); clog << "INFO           {\n";
                         cout << "INFO           {\n";
						 for (int j = 0; j < 4; j++) {
						timeStamp(); clog << "INFO              " << cds->bounds[j][0] << ", " << cds->bounds[j][1] << "\n";
						             cout << "INFO              " << cds->bounds[j][0] << ", " << cds->bounds[j][1] << "\n";
						 }
            timeStamp(); clog << "INFO           }\n";
						  cout << "INFO           }\n";
				  } else { // offline
					  cameraCnt--;
            timeStamp(); clog << "ERROR  Camera " << cds->id << " offline: " << cds->online << ", " << cds->cameraName << ", " << cds->timeStamp << "\n";
						cout << "ERROR  Camera " << cds->id << " offline: " << cds->online << ", " << cds->cameraName << ", " << cds->timeStamp << "\n";
					Globals::hasError = true;

				  }
			  }
      }
  }
}

void Tracker::checkEnterExits() {
	int nObjects;

	ObjectDesc *objectDescs = trax->getObjectDescs(nObjects);

	// exits and enters do not come in order.  We might get an exit and enter for the same tag in one frame so
	// we have to check

	for (int i = 0; i < nObjects; i++, objectDescs++) {
		if(objectDescs->objState == entryState) {
			PersonStats *ps = peopleStats->get(objectDescs->id);
			if (ps == NULL) { // what we want
				ps = new PersonStats(objectDescs->id, Globals::curTime);
				Pattern::theCurPattern->setAvatars(ps, true);
				peopleStats->add(ps);
				worldStats->add(ps);
			} else {
				if(ps->exited) { // got the exit before the enter in the same frame
					worldStats->remove(ps);
					peopleStats->removeAndDestroy(ps);
				} else {
					Pattern::theCurPattern->setAvatars(ps, true);
				}
			}
			// else is was already there and shouldn't be but lets just use it
		} else {
			PersonStats *ps = peopleStats->get(objectDescs->id);
			if(ps == NULL) { // should get the enter in the same frame need to mark
				ps = new PersonStats(objectDescs->id, Globals::curTime);
				ps->exited = true;
				peopleStats->add(ps);
				worldStats->add(ps);
			} else {
				worldStats->remove(ps);
				peopleStats->removeAndDestroy(objectDescs->id);
			}

		}

	}
}

void Tracker::clearTrackingData() {
    timeStamp(); clog << "INFO: Clearing tracking data.\n";
	PersonStats *ps = peopleStats->getHead();
	while(ps != NULL) {
		worldStats->remove(ps);
		peopleStats->removeAndDestroy(ps);
		ps = peopleStats->getHead();
	}
    timeStamp(); clog << "INFO: Tracking data cleared.\n";
}
void Tracker::checkPeople() {
			int nPoints;
			TrackDataBlock *tdb = trax->getTrackDataBlock(nPoints);
			TrackPt *tp = tdb->trackPts;

			for (int i = 0; i < nPoints; i++, tp++)
			{
          PersonStats *ps = peopleStats->get(tp->id);
          if (ps != NULL) {
			  if(useOldCoordCalibration) {
				  float y = (tp->x * scaleY) + offsetY;
				  float x = (tp->y * scaleX) + offsetX;
		           // (x are y are flipped in TYZX tracking system)
				  if(x < cutoffX) {
					  if((x > excludeBoxE) || (y < excludeBoxN) || (y > excludeBoxS)) {
						ps->update(x,y,tp->h);
						worldStats->update(ps);
					  }
				  }
			  } else {
				float x = tp->y; // coordinate systems are flipped
				float y = tp->x; 
				if(bounds->isInBounds(x, y)) {
						ps->update(x,y,tp->h);
						worldStats->update(ps);
				}
			  }
		  } else {
            clog << "WARNING: update data for non existant id.  Creating id: " << tp->id << "\n";
            ps = new PersonStats(tp->id, Globals::curTime);
          peopleStats->add(ps);
          worldStats->add(ps);
          }

		}

}


void Tracker::processTrackData() {
	if (! hasNewData) {
		if (Globals::isOn) {
			if ((worldStats->peopleCnt > 0) &&
			(Globals::curTime > reportModeratorErrorTime)) {
				if(! isModeratorDown) {
					timeStamp(); clog << "ERROR Moderator appears to be down.  (There are people in the space and no moderator updates.)" << endl;
					cout << "ERROR Moderator appears to be down.  (There are people in the space and no moderator updates.)" << endl;
					isModeratorDown = true;
							Globals::hasError = true;

				}
			}
			if (Globals::curTime > reportModeratorEmptyRoomErrorTime) {
				if(! isModeratorDown) {
					timeStamp(); clog << "ERROR Moderator appears to be down.  (The room as appeared empty for " << reportModeratorEmptyRoomErrorInterval << " hours)" << endl;
					cout << "ERROR Moderator appears to be down.  (The room as appeared empty for " << reportModeratorEmptyRoomErrorInterval << " hours)" << endl;
					isModeratorDown = true;
							Globals::hasError = true;

				}
			}
			return;
		}
	}
	
	isModeratorDown = false;
	reportModeratorErrorTime = Globals::curTime + 1000;
	reportModeratorEmptyRoomErrorTime = Globals::curTime + reportModeratorEmptyRoomErrorInterval; 

  checkCameras();
  checkEnterExits();
  checkPeople();
  hasNewData = false;

}

Tracker::~Tracker() {
	clearTrackingData();
	delete trax;
	delete bounds;
}