#include "Recorder.h"
#include "Windows.h"
#include "TrackHash.h"

Recorder::Recorder(string filename, int camCnt, TyzxCam** cams, PersonTrackReceiver *tracker) {
		this->filename = filename;
		this->cams = cams;
		this->tracker = tracker;
		this->camCnt = camCnt;
		zimgFile = new ofstream[camCnt];
		open();
		for(int i = 0; i < camCnt; i++) {
			cams[i]->start();
		}
	}


	void Recorder::open() {
		trackFile.open((filename + "Track.txt").c_str());
		for(int i = 0; i < camCnt; i++) {
			std::stringstream fname;
			fname << filename << "Zimg" << i << ".txt";
			zimgFile[i].open(fname.str().c_str());
		}
	}

	void Recorder::close() {
		trackFile.close();
		for(int i = 0; i < camCnt; i++) {
			zimgFile[i].close();
		}

	}

	void Recorder::run(DWORD duration) {
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
		


	}

	Recorder::~Recorder() {
		close();
	}
