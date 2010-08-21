#include "Recorder.h"
#include "Windows.h"
#include "TrackHash.h"

Recorder::Recorder(string filename, int camCnt, TyzxCam** cams, PersonTrackReceiver *tracker) {
		this->filename = filename;
		this->camCnt = camCnt;
		if(tracker) {
		this->tracker = new  TrackRecorder(tracker);
		this->tracker->open(filename + "Track.txt");
		} else {
			this->tracker = NULL;
		}
		camRecorders = new CamRecorder(camCnt, cams);
		this->camRecorders->open(filename + "Cams.txt");
}



	void CamRecorder::open(string fname) {
		fileStream.open(fname.c_str());
	}
	void TrackRecorder::open(string fname) {
		fileStream.open(fname.c_str());
	}

	void CamRecorder::loop(){
		fileStream << "cnt: " << camCnt << endl;
		for(int i = 0;  i < camCnt; i++) {
			TyzxCam *cam = cams[i];			
			fileStream << "Trn: " << cam->translation.x << " " << cam->translation.y << " " << cam->translation.z << endl;
			fileStream << "Rot: " << cam->rotation.x << " " << cam->rotation.y << " " << cam->rotation.z << endl;
			fileStream << "cxy: " << cam->params.cx << " " << cam->params.cy << endl;
			fileStream << "iUV: " << cam->params.imageCenterU << " " << cam->params.imageCenterV << endl;
		}

		curTime = timeGetTime();
		while(curTime < stopTime) {
			fileStream << curTime << endl;
			for(int i = 0; i < camCnt; i++) {
				unsigned short *zimg = cams[i]->getZImage();
				for(int j = 0; j < cams[i]->imgHeight * cams[i]->imgWidth;j++) {
						fileStream << zimg[j] << " ";
				}
				fileStream << endl;
			}
			fileStream << endl << flush;
			curTime = timeGetTime();

		}
		fileStream.close();
	}
	void TrackRecorder::loop(){
		TrackHash trackhash =  TrackHash();
		curTime = timeGetTime();
		while(curTime < stopTime) {
			tracker->grab(&trackhash);
			fileStream << curTime << endl;
			fileStream <<trackhash.getString() << endl << flush;
			curTime = timeGetTime();

		}
		fileStream.close();
	}

	void Recorder::close() {
		// need to do some sync and change stopTime

	}

	void Recorder::run(DWORD duration) {
		DWORD stopTime = timeGetTime() + duration;

		if(tracker) {
			tracker->setStopTime(stopTime);
		threadCallableTracker tc;
		tc.obj = tracker;
		threadGroup.add_thread(	new boost::thread(tc));
		}
		threadCallableCam tcam;
		tcam.obj = camRecorders; 
		camRecorders->setStopTime(stopTime);

		threadGroup.add_thread(new boost::thread(tcam)	);

		threadGroup.join_all();
//		Sleep(1000); // wait for clean end to threads
		std::cout << "exiting" << std::endl;
		exit(0);

		/*
		TrackHash trackhash =  TrackHash();
		// output image size, reg data, and starttime for each camera

		DWORD startTime = timeGetTime();
		DWORD curTime = startTime;

		for(int i = 0;  i , camCnt; i++) {
			TyzxCam *cam = cams[i];			
			zimgFile[i] << "Trn: " << cam->translation.x << " " << cam->translation.y << " " << cam->translation.z << endl;
			zimgFile[i] << "Rot: " << cam->rotation.x << " " << cam->rotation.y << " " << cam->rotation.z << endl;
			zimgFile[i] << "cxy: " << cam->params.cx << " " << cam->params.cy << endl;
			zimgFile[i] << "iUV: " << cam->params.imageCenterU << " " << cam->params.imageCenterV << endl;
		}


		while(curTime-startTime < duration) {
			for(int i = 0; i < camCnt; i++) {
				TyzxCam *cam = cams[i];
				unsigned short *zimg = cam->getZImage();
				zimgFile[i] << curTime << endl;
				for(int j = 0; j < cam->imgHeight * cam->imgWidth;j++) {
					zimgFile[i] << zimg[j] << " ";
				}
				zimgFile[i] << endl;
			}
			tracker->grab(&trackhash);
			trackFile << curTime << endl;
//			trackFile <<trackhash.getStringStream().rdbuf()->str().c_str() << endl;

			curTime = timeGetTime();
		}
		*/


	}

	Recorder::~Recorder() {
		close();
		delete tracker;
		delete camRecorders;
	
	}
