#ifndef __ELPROPS__
#define __ELPROPS__

//required to get boost program options to link - solution discovered after hours of googling :(
#define BOOST_ALL_DYN_LINK

#define PROP_FPS  "fps"
#define PROP_MINX "worldMinX"
#define PROP_MAXX "worldMaxX"
#define PROP_MINY "worldMinY"
#define PROP_MAXY "worldMaxY"
#define PROP_MINZ "worldMinZ"
#define PROP_MAXZ "worldMaxZ"
#define PROP_PITCH "camPitch"
#define PROP_YAW "camYaw"
#define PROP_ROLL "camRoll"
#define PROP_XOFFSET "camOffsetX"
#define PROP_YOFFSET "camOffsetY"
#define PROP_ZOFFSET "camOffsetZ"
#define PROP_TRACK_WIDTH "trackWidth"
#define PROP_TRACK_HEIGHT "trackHeight"
#define PROP_BG_THRESH "bgThresh"
#define PROP_OSC_ADDRESS "oscAddress"
#define PROP_OSC_PORT "oscPort"
#define PROP_OSC_MINX "oscMinX"
#define PROP_OSC_MAXX "oscMaxX"
#define PROP_OSC_MINZ "oscMinZ"
#define PROP_OSC_MAXZ "oscMaxZ"
#define PROP_SHOW_POINTS "showPointCloud"
#define PROP_SHOW_TRACKS "showTracks"



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

	static void set(string name, string f, bool shouldNotify=true);	
	static void inc(string name, string cat, bool shouldNotify=true); // concat
	static string getString(string name);


	static void set(string name, bool b, bool shouldNotify=true);	
	static void toggle(string name, bool shouldNotify=true);
	static bool getBool(string name);

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
