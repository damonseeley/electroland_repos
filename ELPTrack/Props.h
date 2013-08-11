#ifndef __ELPROPS__
#define __ELPROPS__

//required to get boost program options to link - solution discovered after hours of googling :(
#define BOOST_ALL_DYN_LINK


#define PROP_FPS  "fps"
#define PROP_MINX "minX"
#define PROP_MAXX "maxX"
#define PROP_MINY "minY"
#define PROP_MAXY "maxY"
#define PROP_MINZ "minZ"
#define PROP_MAXZ "maxZ"
#define PROP_PITCH "pitch"
#define PROP_YAW "yaw"
#define PROP_ROLL "roll"
#define PROP_XOFFSET "xOffset"
#define PROP_YOFFSET "yOffset"
#define PROP_ZOFFSET "zOffset"
#define PROP_TRACK_WIDTH "trackWidth"
#define PROP_TRACK_HEIGHT "trackHeight"
#define PROP_BG_THRESH "bgThresh"

#include <string>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <map>
#include <vector>
#include "PropChangeListener.h"
using namespace std;
namespace po = boost::program_options;

class Props {
public:


	/*
	 bool is_empty(const boost::any & operand); 
     bool is_int(const boost::any & operand); 
     bool is_double(const boost::any & operand); 
     bool is_char_ptr(const boost::any & operand); 
     bool is_string(const boost::any & operand); 
	 */

	static Props *theProps;
	static void initProps(int argc, char** argv);
	static void writeToFile(string filename);

	static void set(string name, boost::any value, bool shouldNotify=true);	
	static boost::any get(string name);
	
	static void set(string name, float f, bool shouldNotify=true);	
	static void inc(string name, float amount, bool shouldNotify=true);
	static float getFloat(string name);


	static void set(string name, int f, bool shouldNotify=true);	
	static void inc(string name, int amount, bool shouldNotify=true);
	static int getInt(string name);

	static void notifyListeners();

	
private:
	std::vector<PropChangeListener> changeListeners;
	std::map<string, boost::any> map;

	po::options_description optionDesc;
	po::variables_map optionVM;


	void init(int argc, char** argv);

	Props();
	
	~Props(){}

	

};


#endif
