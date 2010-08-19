# ifndef __PERSON_TRACK_RECEIVER_H__
# define __PERSON_TRACK_RECEIVER_H__


#include "PersonTrackAPI.h"
#include "TrackHash.h"

#include <string>
using namespace std;

class PersonTrackReceiver
{
public:
	long curFrame;

	float scale;

	float PixelsPerUnit	;
	float LeftCoord	;
	float Width	;
	float BottomCoord ;
	float Height;
	PersonTrackAPI::tyzxUnits Units;
		int	stat;



	PersonTrackAPI *trax;


	PersonTrackReceiver(string moderatorIP);
	void start();
	void grab(TrackHash* hash);
	~PersonTrackReceiver();
	static  bool myWarningErrorFun(const char	*message);
	static void myFatalErrorFun(const char *message);
	void setScale(float s) { scale = s; }

}
;
# endif
