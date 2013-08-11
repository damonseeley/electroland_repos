#include "Props.h"
#include <fstream>
#include <iostream>
//#include <locale>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>

Props* Props::theProps = NULL;


Props::Props() {
}




void Props::writeToFile(string filename) {
	ofstream file;
	file.open(filename);

  time_t rawtime;
  struct tm * timeinfo;
  char buffer [80];

  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  strftime (buffer,80,"%Y-%m-%d-%H-%M-%S",timeinfo);

	file << "# automatically generated property file" << std::endl;
	file << "# " << buffer << std::endl;
	file << std::endl;

	for (std::map<string, boost::any>::iterator it=theProps->map.begin(); it != theProps->map.end(); it++ ) { 
		   if(typeid(float) == it->second.type()) {
		   file << it->first << "=" << boost::any_cast<float>(it->second) <<std::endl;
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
		("file,f", po::value<string>(&configFileName)->default_value("track.ini"), "optional init file.  If not specified \'track.ini\' is used")
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
		(PROP_BG_THRESH, po::value<float>()->default_value(.075),  " background model theshold")
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

