#ifndef __SOUNDPLAYER_H__
#define __SOUNDPLAYER_H__

#include <stdio.h> // a sztringmûveletekhez
#include "dsutil.h" // a directsoundhoz

class SoundPlayer {
  
  CSoundManager* g_pSoundManager = NULL;
  CSound*        g_pSound = NULL;
  BOOL           g_bBufferPaused;

public:

  SoundPlayer();
  ~SoundPlayer();
}
;

#endif
