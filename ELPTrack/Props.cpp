#include "Props.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

//#include <locale>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>

Props* Props::theProps = NULL;


Props::Props() {
}



	
void Props::writeToFile(string filename) {

	int versionNum = 0;
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

	file << "# automatically generated ELPTrack property file" << std::endl;
	file << "# " << buffer << std::endl;

	for (std::map<string, boost::any>::iterator it=theProps->map.begin(); it != theProps->map.end(); it++ ) { 
		   if(typeid(float) == it->second.type()) {
		   file << setw(10) << std::left << it->first << "= " << setw(10) << boost::any_cast<float>(it->second) <<
			    "# " << (theProps->optionDesc.find(it->first, false)).description() <<std::endl;
		   } else if (typeid(int) == it->second.type()) {
		   file << setw(10) << std::left << it->first << "= " << setw(10) << boost::any_cast<int>(it->second) <<
			    "# " << (theProps->optionDesc.find(it->first, false)).description() <<std::endl;
		   } else if (typeid(string) == it->second.type()) {
		   file << setw(10) << std::left << it->first << "= " << setw(10) << boost::any_cast<string>(it->second) <<
			    "# " << (theProps->optionDesc.find(it->first, false)).description() <<std::endl;
		   } else if (typeid(bool) == it->second.type()) {
		   file << setw(10) << std::left << it->first << "= " << setw(10) << boost::any_cast<bool>(it->second) <<
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



void Props::initProps(int argc, char** argv) {
	if(theProps) return;
	theProps = new Props();
	theProps->init(argc, argv);
}

void Props::init(int argc, char** argv) {
	string configFileName;

	optionDesc.add_options()
		("help,h", "displays this help message")
		("file,f", po::value<string>(&configFileName)->default_value("ELPTrack.ini"), "optional init file.  If not specified \'ELPTrack.ini\' is used")
		(PROP_FPS,  po::value<float>()->default_value(20.0f), "maximum for tracking")
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
		(PROP_TRACK_WIDTH, po::value<int>()->default_value(60),  "width of plan view image (tracking percision is maxX-minX/width)")
		(PROP_TRACK_HEIGHT, po::value<int>()->default_value(120),  "height of plan view image (tracking percision is maxZ-minZ/height)")
		(PROP_BG_THRESH, po::value<float>()->default_value(.075),  "background model theshold")
		(PROP_OSC_ADDRESS, po::value<string>()->default_value(""), "IP address of OSC receiver (an empty string will not send msgs)")
		(PROP_OSC_PORT, po::value<int>()->default_value(7000), "port of OSC receiver")
		(PROP_OSC_MINX, po::value<float>()->default_value(0), "min x value for tracks sent over osc")
		(PROP_OSC_MINZ, po::value<float>()->default_value(0), "min z value for tracks sent over osc")
		(PROP_OSC_MAXX, po::value<float>()->default_value(1.0f), "max x value for tracks sent over osc")
		(PROP_OSC_MAXZ, po::value<float>()->default_value(1.0f), "max z value for tracks sent over osc")
		(PROP_SHOW_POINTS, po::value<bool>()->default_value(true), "show the point cloud for interactive configuration")
		(PROP_SHOW_TRACKS, po::value<bool>()->default_value(true), "show the tracks")
		;
//todo flip x,y,z
	try {
		po::store(po::parse_command_line(argc, argv, optionDesc), optionVM);


		if(optionVM.count("help")) {
			std::cout << optionDesc << std::endl;
			exit(0);
		}
		optionVM.notify();
	} catch(po::error& e) {
		std::cerr << "ERROR in Command line arguments: " << e.what() << std::endl;
		std::cerr << optionDesc << std::endl;
		exit(1);
	}

	try {
		std::ifstream configFileStream(configFileName);
		if(configFileStream) {
			po::store(po::parse_config_file(configFileStream, optionDesc, false), optionVM);


			if(optionVM.count("help")) {
				std::cout << optionDesc << std::endl;
				exit(0);
			}
		}

	} catch(po::error& e) {
		std::cerr << "ERROR in config file: " << e.what() << std::endl;
		std::cerr << optionDesc << std::endl;
		exit(1);
	}
	for (po::variables_map::iterator it=optionVM.begin(); it != optionVM.end(); it++ ) { 
		//map.insert(std::pair<string, boost::any>(it->first,boost::any(it->second.value()))); 
		map[it->first] = boost::any(it->second.value());

	}
//	map.insert(optionVM.begin(), optionVM.end());
}


//void Props::set( const std::string& opt, const float& val)
//{
 // optionVM[opt].value() = boost::any(val);
//}

