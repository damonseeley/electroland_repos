#ifndef __ELMESA_CAM__
#define __ELMESA_CAM__

//needed for mesa to include mesa
#define NOMINMAX
#include <Windows.h> // only on a Windows system
#undef NOMINMAX

#include <libMesaSR.h>
#include <opencv2/core/core.hpp>


class MesaCam {
public:
	SRCAM srCam; // SwissRanger

	int camMode;


	MesaCam();

	bool open(const char *filenameOrAddress, bool isIP);

	bool aquire();

	bool aquireRange(cv::Mat &img);
	bool aquireIntensity(cv::Mat &img);
	bool aquireConfMap(cv::Mat &img);

	cv::Mat getRangeImage();

	bool aquire(cv::Mat &range, cv::Mat &intesity);
	~MesaCam();

	void setIntergrationTime(float ms);
	void setDualIntergrationTime(int ratio);
	void setAmpThresh(unsigned short);
	void useAutoExposure(bool b);
	void setTimeout(int ms);
	void setModulationFrequency(int f);

	void setFixPatternNoise(bool b);
	void setUseMedianFilter(bool b);
	void setConvertGray(bool b);
	void setGenConfMap(bool b);
	void setUseAdptiveFilter(bool b);
	void setUseNonAmbiguityMode(bool b);
	void setMode();


	void setupCameraFromProps();
	void setPropsFromCam();

};

#endif