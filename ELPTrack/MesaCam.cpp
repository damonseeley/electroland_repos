#include "MesaCam.h"
#include "ErrorLog.h"
#include "Props.h"

MesaCam::MesaCam() {
	//suggested by mesa
	camMode = AM_COR_FIX_PTRN|AM_CONV_GRAY|AM_DENOISE_ANF ;

}

bool MesaCam::open(const char *filenameOrAddress, bool isIP) {
	int success = 0;

	if(isIP) {
		success = SR_OpenETH(&srCam, filenameOrAddress);
	} else {
		if(strcmp("dialog", filenameOrAddress) == 0) {
			success = SR_OpenDlg(&srCam, 2, 0); // 2 -> opens a selection dialog
		} else {
			success = SR_OpenFile(&srCam, filenameOrAddress);
		}
	}

	if(success < 0) {
		*ErrorLog::log << "Unable to open mesa camera " << filenameOrAddress << ". Mesa error "<< success <<std::endl;
		return false;
	} else {
		return true;
	}
}

MesaCam::~MesaCam() {
	int result = SR_Close(srCam);
	switch(result) {
	case 0:
		//success
		break;
	case -1:
		*ErrorLog::log << "Unable to close Mesa camera.  Mesa error -1 \'wrong device\'" << std::endl;
		break;
	case -2:
		*ErrorLog::log << "Unable to close Mesa camera.  Mesa error -2 \'can\'t release interface\'" << std::endl;
		break;
	case -3:
		*ErrorLog::log << "Unable to close Mesa camera.  Mesa error -3 \'can\'t close device\'" << std::endl;
		break;
	default:
		*ErrorLog::log << "Mesa camera may not be closed correctly.  SR_Close returned " << result << " which is undefined in the API" << std::endl;
		break;

	}

}

bool MesaCam::aquire() {
	return SR_Acquire(srCam) >= 0;
}


cv::Mat MesaCam::getRangeImage() {
	cv::Size imsize(SR_GetCols(srCam), SR_GetRows(srCam)); // SR image size
	return cv::Mat(imsize, CV_16UC1, SR_GetImage(srCam, 0));
}

cv::Mat MesaCam::getIntensityImage() {
	cv::Size imsize(SR_GetCols(srCam), SR_GetRows(srCam)); // SR image size
	return cv::Mat(imsize, CV_16UC1, SR_GetImage(srCam, 1));
}

cv::Mat MesaCam::getConfMap() {
	cv::Size imsize(SR_GetCols(srCam), SR_GetRows(srCam)); // SR image size
	return cv::Mat(imsize, CV_16UC1, SR_GetImage(srCam, 2));
}

/*
bool MesaCam::aquireRange(cv::Mat &img) {
	if(SR_Acquire(srCam) < 0)
		return false;
	cv::Size imsize(SR_GetCols(srCam), SR_GetRows(srCam)); // SR image size
	cv::Mat(imsize, CV_16UC1, SR_GetImage(srCam, 0)).copyTo(img);
	return true;
}

bool MesaCam::aquireIntensity( cv::Mat &img) {
	if(SR_Acquire(srCam) < 0)
		return false;
	cv::Size imsize(SR_GetCols(srCam), SR_GetRows(srCam)); // SR image size
	cv::Mat(imsize, CV_16UC1, SR_GetImage(srCam, 1)).copyTo(img);
	return true;
}

bool MesaCam::aquire(cv::Mat &range, cv::Mat &intesity) {
	if(SR_Acquire(srCam) < 0) 
		return false;
	cv::Size imsize(SR_GetCols(srCam), SR_GetRows(srCam)); // SR image size
	cv::Mat(imsize, CV_16UC1, SR_GetImage(srCam, 0)).copyTo(range);
	cv::Mat(imsize, CV_16UC1, SR_GetImage(srCam, 1)).copyTo(intesity);
	return true;
}
bool MesaCam::aquireConfMap(cv::Mat &img) {
	//assumes the mode is set correctly
	if(SR_Acquire(srCam) < 0)
		return false;
	cv::Size imsize(SR_GetCols(srCam), SR_GetRows(srCam)); // SR image size
	cv::Mat(imsize, CV_16UC1, SR_GetImage(srCam, 2)).copyTo(img);
	return true;
}
*/
void MesaCam::setModulationFrequency(int f) {
	if(f < 0) return;
	ModulationFrq frq;
	switch(f) {
	case 40:
		frq = MF_40MHz ;
		break;
	case 30:
		frq = MF_30MHz ;
		break;
	case 21:
		frq = MF_21MHz ;
		break;
	case 20:
		frq = MF_20MHz ;
		break;
	case 19:
		frq = MF_19MHz ;
		break;
	case 15:
		frq = MF_15MHz ;
		break;
	case 29:
		frq = MF_29MHz ;
		break;
	case 31:
		frq = MF_31MHz ;
		break;
	case 145:
		frq = MF_14_5MHz ;
		break;
	case 155:
		frq = MF_15_5MHz ;
		break;
	case 10:
		std::cout << "WARNING: A modulation frequence of 10 may not be supported by the API" << std::endl;
		frq = (ModulationFrq) (12);
		break;

	}

	 SR_SetModulationFrequency(srCam, frq);
}

// valid range seems to be betwwne .3 and 25.8
void MesaCam::setIntergrationTime(float ms) {
	//from api docs ms = 0.300ms+(intTime)*0.100 ms
	int intTime = (int) ((ms-.3) * 10); 
	if(intTime <0) {
		intTime = 0;
		*ErrorLog::log << "Intergration time is out of range (permitted values are .3-25.8) using .3" << std::endl;	
	}
	if(intTime > 255) {
		intTime = 255;
		*ErrorLog::log << "Intergration time is out of range (permitted values are .3-25.8) using 25.8" << std::endl;	
	}
	SR_SetIntegrationTime  ( srCam, intTime);
}


void MesaCam::setAmpThresh(unsigned short ampThesh) {
	SR_SetAmplitudeThreshold(srCam, ampThesh);
}

void MesaCam::useAutoExposure(bool b) {
	if(b) {
		// for SR4k good values are 1,150,5,70 
		SR_SetAutoExposure(srCam, 1,150,5,70);
	} else {
		// minIntTime=0xff the AutoExposure is turned off.
		SR_SetAutoExposure(srCam, 0xff,0,0,0);
	}
}
void MesaCam::setTimeout(int ms) {
	if(ms < 0) return;
	SR_SetTimeout(srCam, ms);
}


void MesaCam::setDualIntergrationTime(int ratio) {
	if(ratio < 0) return;
	ratio = (ratio > 100) ? 100 : ratio;
	SR_SetDualIntegrationTime (srCam, ratio);
}


void MesaCam::setFixPatternNoise(bool b) {
	if(b) {
		camMode |= AM_COR_FIX_PTRN ;
	} else {
		camMode &= ~AM_COR_FIX_PTRN;
	}
}
void MesaCam::setUseMedianFilter(bool b) {
	if(b) {
		camMode |= AM_MEDIAN  ;
	} else {
		camMode &= ~AM_MEDIAN ;
	}
}
void MesaCam::setConvertGray(bool b){ 
	if(b) {
		camMode |= AM_CONV_GRAY   ;
	} else {
		camMode &= ~AM_CONV_GRAY  ;
	}
}
void MesaCam::setGenConfMap(bool b) {
	if(b) {
		camMode |= AM_CONF_MAP    ;
	} else {
		camMode &= ~AM_CONF_MAP   ;
	}
}
void MesaCam::setUseAdptiveFilter(bool b) {
	if(b) {
		camMode |= AM_DENOISE_ANF     ;
	} else {
		camMode &= ~AM_DENOISE_ANF    ;
	}
}
void MesaCam::setUseNonAmbiguityMode(bool b) {
	if(b) {
		camMode |=AM_NO_AMB     ;
	} else {
		camMode &= ~AM_NO_AMB    ;
	}
}
void MesaCam::setMode() {
	SR_SetMode  (srCam,  camMode);
} 

void MesaCam::setupCameraFromProps() {
	setIntergrationTime(Props::getFloat(PROP_MESA_INT_TIME));
	setDualIntergrationTime(Props::getInt(PROP_MESA_DUAL_INT_TIME));
	setAmpThresh(Props::getInt(PROP_MESA_AMP_THRESH));
	setTimeout(Props::getInt(PROP_MESA_TIMEOUT));
	//set auto exposure AFTER setting intergraton time
	// to overide other setting
	useAutoExposure(Props::getBool(PROP_MESA_AUTOEXP));

	setModulationFrequency(Props::getInt(PROP_MESA_MODFREQ));
	//the following are set via setMode
	setFixPatternNoise(Props::getBool(PROP_MESA_PAT_NOISE));
	setUseMedianFilter(Props::getBool(PROP_MESA_AM_MEDIAN));
	setConvertGray(Props::getBool(PROP_MESA_CONV_GRAY));
	setGenConfMap(Props::getBool(PROP_MESA_GEN_CONF_MAP));
	setUseAdptiveFilter(Props::getBool(PROP_MESA_DENOISE));
	setUseNonAmbiguityMode(Props::getBool(PROP_MESA_NONAMBIG));
	setMode();
	
}

void MesaCam::setPropsFromCam() {
	
	Props::set(PROP_MESA_INT_TIME, 0.300f+((float) SR_GetIntegrationTime(srCam))*0.100f);
//	Props::set(PROP_MESA_DUAL_INT_TIME, (int) SR_GetDualIntegrationTime(srCam)); missing from api
	Props::set(PROP_MESA_AMP_THRESH, SR_GetAmplitudeThreshold(srCam));
//	Props::set(PROP_MESA_TIMEOUT, SR_GetTimeout(srCam)); missing from api
	
	int mode = SR_GetMode(srCam);
	Props::set(PROP_MESA_PAT_NOISE, (AM_COR_FIX_PTRN & mode) != 0);
	Props::set(PROP_MESA_AM_MEDIAN, (AM_MEDIAN  & mode) != 0);
	Props::set(PROP_MESA_CONV_GRAY, (AM_CONV_GRAY & mode) != 0);
	Props::set(PROP_MESA_GEN_CONF_MAP, (AM_CONF_MAP & mode) != 0);
	Props::set(PROP_MESA_DENOISE, (AM_DENOISE_ANF  & mode) != 0);
	Props::set(PROP_MESA_NONAMBIG, (AM_NO_AMB & mode) != 0);

	ModulationFrq frq = SR_GetModulationFrequency  (srCam);
	switch(frq) {
	case MF_40MHz:
		Props::set(PROP_MESA_MODFREQ, 40);
	case MF_30MHz:
		Props::set(PROP_MESA_MODFREQ, 30);
		break;
	case MF_21MHz:
		Props::set(PROP_MESA_MODFREQ, 21);
		break;
	case MF_20MHz:
		Props::set(PROP_MESA_MODFREQ, 20);
		break;
	case MF_19MHz:
		Props::set(PROP_MESA_MODFREQ, 19);
		break;
	case MF_15MHz:
		Props::set(PROP_MESA_MODFREQ, 15);
		break;
	case MF_29MHz:
		Props::set(PROP_MESA_MODFREQ, 29);
		break;
	case MF_31MHz:
		Props::set(PROP_MESA_MODFREQ, 31);
		break;
	case MF_14_5MHz:
		Props::set(PROP_MESA_MODFREQ, 14);
		break;
	case MF_15_5MHz:
		Props::set(PROP_MESA_MODFREQ, 15);
		break;
	case 12: // not sure if this will work
		Props::set(PROP_MESA_MODFREQ, 10);
		break;
 
 

	}
}


