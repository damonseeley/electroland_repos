#include "Props.h"
#include "ErrorLog.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
//#include <locale>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include "Shlwapi.h"

Props* Props::theProps = NULL;

std::string Props::curDir = "" ;
std::string Props::version = "";

Props::Props() {	
}



	
void Props::writeToFile(string filename) {
	
	int versionNum = 0;
	
	
	// prepend dir cached at startup since mesa changes working directory
	if(PathIsRelative(filename.c_str())) {
		filename = curDir + "\\" + filename;
	} 

	string freeFilename = filename + ".ini";

	while(FILE *testFile = fopen(freeFilename.c_str(), "r")) {
		std::cout << freeFilename << " is already in use." << std::endl;
		fclose(testFile);
		stringstream  ss;
		ss << filename << "_" <<  versionNum++ << ".ini";
		freeFilename = ss.str(); 

	}
	std::cout << " saving properties to " << freeFilename << std::endl;
	ofstream file;
	file.open(freeFilename);

  time_t rawtime;
  struct tm * timeinfo;
  char buffer [80];

  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  strftime (buffer,80,"%Y-%m-%d, %H-%M-%S",timeinfo);
	file << "# ELPTrack " << version << std::endl;
	file << "# automatically generated property file" << std::endl;
	file << "# " << buffer << std::endl;
	file << "# " << std::endl << std::endl;
	for (std::map<string, boost::any>::iterator it=theProps->map.begin(); it != theProps->map.end(); it++ ) { 
		   if(typeid(float) == it->second.type()) {
		   file << setw(10) << std::left << it->first << "= " << setw(20) << boost::any_cast<float>(it->second) <<
			    "# " << (theProps->optionDesc.find(it->first, false)).description() <<std::endl;
		   } else if (typeid(int) == it->second.type()) {
		   file << setw(10) << std::left << it->first << "= " << setw(20) << boost::any_cast<int>(it->second) <<
			    "# " << (theProps->optionDesc.find(it->first, false)).description() <<std::endl;
		   } else if (typeid(string) == it->second.type()) {
		   file << setw(10) << std::left << it->first << "= " << setw(20) << boost::any_cast<string>(it->second) <<
			    "# " << (theProps->optionDesc.find(it->first, false)).description() <<std::endl;
		   } else if (typeid(bool) == it->second.type()) {
		   file << setw(10) << std::left << it->first << "= " << setw(20) << boost::any_cast<bool>(it->second) <<
			    "# " << (theProps->optionDesc.find(it->first, false)).description() <<std::endl;
		   }
	} 
	file.close();
}


void Props::set(string name, boost::any value, bool shouldNotify) {
	Props::theProps->map[name] = value;
	if(shouldNotify) {
		notifyListeners();
	}
}

//floats
void Props::inc(string name, float amount, bool shouldNotify) {
	set(name, getFloat(name) + amount, shouldNotify);
}

void Props::set(string name, float value, bool shouldNotify) {
	set(name, boost::any(value), shouldNotify);
}
float Props::getFloat (string name) {
	return boost::any_cast<float>(get(name));
}

//ints
void Props::inc(string name, int amount, bool shouldNotify) {
	set(name, getInt(name) + amount, shouldNotify);
}

void Props::set(string name, int value, bool shouldNotify) {
	set(name, boost::any(value), shouldNotify);
}
int Props::getInt (string name) {
	return boost::any_cast<int>(get(name));
}


//strings
void Props::inc(string name, string amount, bool shouldNotify) {
	set(name, getString(name) + amount, shouldNotify); // concat
}

void Props::set(string name, string value, bool shouldNotify) {
	set(name, boost::any(value), shouldNotify);
}
string Props::getString (string name) {
	return boost::any_cast<string>(get(name));
}

//bool 
void Props::set(string name, bool b, bool shouldNotify) {
	set(name, boost::any(b), shouldNotify);
}
void Props::toggle(string name, bool shouldNotify) {
	set(name, ! getBool(name), shouldNotify); // concat
}
bool Props::getBool(string name) {
	return boost::any_cast<bool>(get(name));
}

boost::any Props::get(string name) {
	return Props::theProps->map[name];
}
void Props::notifyListeners(){
	for(std::vector<PropChangeListener>::iterator it = Props::theProps->changeListeners.begin(); it != Props::theProps->changeListeners.end(); it++) {
		it->propsChanged();
	}
}



void Props::initProps(int argc, char** argv, string version) {
	if(theProps) return;
	Props::version = version;
	theProps = new Props();
		char cCurrentPath[FILENAME_MAX];

	 if (!_getcwd(cCurrentPath, sizeof(cCurrentPath)))
		 {
			 *ErrorLog::log << "Props unable to discover current working directory" << std::endl;
		 }
		cCurrentPath[sizeof(cCurrentPath) - 1] = '\0'; /* probably not really required but lets be safe*/
		curDir = std::string(cCurrentPath);


	theProps->init(argc, argv);
}

void Props::init(int argc, char** argv) {
	string configFileName;

	optionDesc.add_options()
		("help,h", "displays this help message when passed in on the command line")
		("version,v", "displays the program version string")
		(PROP_FILE ",f", po::value<string>(&configFileName)->default_value("ELPTrack.ini"), "Path to config file.  If not specified \'ELPTrack.ini\' is used")
		(PROP_ERROR_LOG, po::value<string>()->default_value("ELPTrackError.log"), "Path to log file.  If not specified \'ELPTrackError.log\' is used")
		(PROP_FPS,  po::value<float>()->default_value(30.0f), "maximum for tracking")
		(PROP_BADFRAMES,  po::value<int>()->default_value(4), "number of consecutive bad acquires permitted before exiting (if the camera is down each frame takes ~20sec)")
		(PROP_MINX, po::value<float>()->default_value(-3.0f), "minimum x value (in m) for tracking")
		(PROP_MAXX, po::value<float>()->default_value(3.0f),  "maximum x value (in m) for tracking")
		(PROP_MINY, po::value<float>()->default_value(0.0f), "minimum y value (in m) for tracking")
		(PROP_MAXY, po::value<float>()->default_value(2.0f),  "maximum y value (hieght in m) for tracking")
		(PROP_MINZ, po::value<float>()->default_value(0.0f), "minimum z value (in m) for tracking")
		(PROP_MAXZ, po::value<float>()->default_value(10.0f),  "maximum z value (in m) for tracking")
		(PROP_PITCH, po::value<float>()->default_value(0.0f),  "camera pitch")
		(PROP_YAW, po::value<float>()->default_value(0.0f),  "camera yaw")
		(PROP_ROLL, po::value<float>()->default_value(0.0f),  "camera roll")
		(PROP_XOFFSET, po::value<float>()->default_value(0.0f),  "camera x offset")
		(PROP_YOFFSET, po::value<float>()->default_value(0.0f),  "camera y offset")
		(PROP_ZOFFSET, po::value<float>()->default_value(0.0f),  "camera z offset")
		
		(PROP_PLANVIEW_WIDTH, po::value<int>()->default_value(60),  "width of plan view image (tracking percision is maxX-minX/width)")
		(PROP_PLANVIEW_HEIGHT, po::value<int>()->default_value(120),  "height of plan view image (tracking percision is maxZ-minZ/height)")
		(PROP_PLANVIEW_THRESH, po::value<float>()->default_value(2.0),  "number of pionts needed per grid cell")
		(PROP_PLANVIEW_BLUR_R, po::value<int>()->default_value(0), "Size of blur kernal used to planview before thresholding.  Should be an odd value.  Less than 3 == no blur")
		(PROP_PLANVIEW_FLIPX, po::value<bool>()->default_value(false), "flip track\'s x coordinates")
		(PROP_PLANVIEW_FLIPZ, po::value<bool>()->default_value(true), "flip track\'s z coordinates")
		
		(PROP_BG_THRESH, po::value<float>()->default_value(.075f),  "background model theshold")
		(PROP_BG_ADAPT, po::value<float>()->default_value(.001f), "background adaptation rate (must be in range 0-1.  0 will use first image, 1 will use last frame")
		(PROP_OSC_ADDRESS, po::value<string>()->default_value(""), "IP address of OSC receiver (an empty string will not send msgs)")
		(PROP_OSC_PORT, po::value<int>()->default_value(7000), "port of OSC receiver")
		(PROP_OSC_MINX, po::value<float>()->default_value(0), "min x value for tracks sent over osc, set min & max to 0 to use world coords")
		(PROP_OSC_MINZ, po::value<float>()->default_value(0), "min z value for tracks sent over osc, set min & max to 0 to use world coords")
		(PROP_OSC_MAXX, po::value<float>()->default_value(1.0f), "max x value for tracks sent over osc, set min & max to 0 to use world coords")
		(PROP_OSC_MAXZ, po::value<float>()->default_value(1.0f), "max z value for tracks sent over osc, set min & max to 0 to use world coords")
		(PROP_SHOW_POINTS, po::value<bool>()->default_value(true), "show the point cloud for interactive configuration")
		(PROP_SHOW_TRACKS, po::value<bool>()->default_value(true), "show the tracks")
		(PROP_SHOW_RANGE, po::value<bool>()->default_value(true), "show the mesa range image")
		(PROP_SHOW_GRAY, po::value<bool>()->default_value(true), "show the mesa gray image")
		(PROP_SHOW_BGSUB, po::value<bool>()->default_value(true), "show the range image after background subtraction")
		(PROP_TRACK_PROVISIONAL_TIME, po::value<int>()->default_value(1000), "ms after a track appears before it is considered established")
		(PROP_TRACK_TIME_TO_DIE, po::value<int>()->default_value(1000), "ms after a track is lost before it is removed")
		(PROP_TRACK_TIME_TO_DIE_PROV, po::value<int>()->default_value(500), "ms after a provisional track is lost before it is remmoved")
		(PROP_TRACK_SMOOTHING, po::value<float>()->default_value(.75f), "smoothing of track movement (range is 0-1)")
		(PROP_TRACK_MAX_MOVE, po::value<float>()->default_value(9), "Maximum units in plan view a track can move per frame.")
		(PROP_PRINT_TRACK_TO_CONSOLE, po::value<bool>()->default_value(false), "Print tracks to console.")
		(PROP_MESA_CAM, po::value<string>()->default_value("dialog"), "Mesa data source (filename, ip address, or \'dialog\').")
		(PROP_MESA_INT_TIME, po::value<float>()->default_value(3.3f), "Mesa Dual integration time of the camera.The ratio is a value from 0 to 100. -1 leave value unchanged")
		(PROP_MESA_DUAL_INT_TIME, po::value<int>()->default_value(0), "mesa dual intergration ratio (0 turns it off)")
		(PROP_MESA_AMP_THRESH, po::value<int>()->default_value(0), "mesa amplitude threshold (range 0-(2^16)-1)")
		(PROP_MESA_AUTOEXP, po::value<bool>()->default_value(false), "use autoexposure with recomended values to set intergration time instead of " PROP_MESA_INT_TIME)
		(PROP_MESA_TIMEOUT, po::value<int>()->default_value(-1), "timeout for mesa operations in ms (-1 -> leaves the value unchanged)")
		(PROP_MESA_PAT_NOISE, po::value<bool>()->default_value(true), "fix pattern noise correction (the mesa docs say it should always be on)")
		(PROP_MESA_AM_MEDIAN, po::value<bool>()->default_value(false), "turns on a 3x3 median filter")
		(PROP_MESA_CONV_GRAY, po::value<bool>()->default_value(false), "adjust the amplitude image to look more like a conventional greyscale image")
		(PROP_MESA_GEN_CONF_MAP, po::value<bool>()->default_value(false), "generate confidence maps")
		(PROP_MESA_DENOISE, po::value<bool>()->default_value(true), "turns on the 5x5 hardware adaptive neighborhood filter")
		(PROP_MESA_NONAMBIG, po::value<bool>()->default_value(true), "turns on non ambiguity mode ")
		(PROP_MESA_MODFREQ, po::value<int>()->default_value(10), "Set Mesa modulation frequency.  Expected values 10, 155 (for 15.5Mhz), 145 (for 14.5mhz), 31, 29, 15, 19, 20, 21, 30, or 40.s")
		(PROP_USE_MESA_CAM_SETTINGS, po::value<bool>()->default_value(true), "use mess settings on camera (ignore settings in this config file)")




				;
//todo flip x,y,z
	try {
		po::store(po::parse_command_line(argc, argv, optionDesc), optionVM);


		if(optionVM.count("help")) {
			std::cout << optionDesc << std::endl;
			exit(0);
		}
		if(optionVM.count("version")) {
			std::cout << version << std::endl;
			exit(0);
		}
		optionVM.notify();
	} catch(po::error& e) {
		*ErrorLog::log << "ERROR in Command line arguments: " << e.what() << std::endl;
		std::cerr << optionDesc << std::endl;
		exit(1);
	}

	try {
		std::ifstream configFileStream(configFileName);
		if(configFileStream) {
			po::store(po::parse_config_file(configFileStream, optionDesc, false), optionVM);
		} else {
			*ErrorLog::log  << "CONFIGURATION FILE " << configFileName << " NOT FOUND.***  Using default values" << std::endl;
		}

	} catch(po::error& e) {
		std::cerr << "ERROR IN CONFIGURATION FILE: " << e.what() << std::endl;
		std::cerr << std::endl << optionDesc << std::endl << std::endl;
		*ErrorLog::log << "ERROR IN CONFIGURATION FILE: " << e.what() << std::endl;
		exit(1);
	}

	for (po::variables_map::iterator it=optionVM.begin(); it != optionVM.end(); it++ ) { 
		map[it->first] = boost::any(it->second.value());

	}
//	map.insert(optionVM.begin(), optionVM.end());
}


void niceExit(int value) {
//		timeEndPeriod(1);
		exit(value);
}

//void Props::set( const std::string& opt, const float& val)
//{
 // optionVM[opt].value() = boost::any(val);
//}

