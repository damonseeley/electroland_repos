#include "WorldStats.h"

WorldStats::WorldStats() { 
	reportStatsInterval = CProfile::theProfile->Int("reportStatsInterval", 15) * 1000 * 60;
	peopleCnt=0;
reportStatsTime = 0;
frames = 0;
  peopleSum = 0;
  maxPeople = 0;
  minPeople = INT_MAX;
  // per interval stats

}

void WorldStats::reset() {
	peopleCnt=0;
reportStatsTime = 0;
frames = 0;
  peopleSum = 0;
  maxPeople = 0;
  minPeople = INT_MAX;
}
void WorldStats::update(int curTime, int deltaTime) {
	frames++;
	peopleSum +=peopleCnt;
	maxPeople = (peopleCnt > maxPeople) ? peopleCnt : maxPeople;
	minPeople = (peopleCnt < minPeople) ? peopleCnt : minPeople;
	if(Globals::curTime >= reportStatsTime) {
		reportStats();
	}
}

void WorldStats::reportStats() {
	if(frames > 0) {
		timeStamp(); clog << 
			"STATS  AVE=" << (float) peopleSum / (float) frames <<
			"  MAX=" << maxPeople <<
			"  MIN=" << minPeople << "\n";
	}
  frames = 0;
  peopleSum = 0;
  maxPeople = 0;
  minPeople = INT_MAX;
  reportStatsTime = Globals::curTime + reportStatsInterval;
}
