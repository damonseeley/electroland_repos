ELPTrack v1.0b5 (Release) 10/25/2013
changed track IDs to wrap around to 0 instead of -LONG_MAX
exposed all relavent blob detection parameters see http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html#simpleblobdetector
checked step size of t/T for background subtraction thresh
removed planViewThresh and replaced with planViewUpperThresh
blobDetection now works with gray scale image

ELPTrack v1.0b4 (Release) 10/03/2013
fixed bug with backgournd subtraction not removing points in cloud
added catch all try/catch block  e.g. catch(...)
added try/catch blocks to display thread

ELPTrack v1.0b3 (Release) 9/29/2013
forgot to change the version number
added try/catch blocks around everything in an effort to locate an intermittend crash and keep the application running

ELPTrack v1.0b3 (Release) 9/29/2013
Added planViewBlurSize to increase tracking quality

ELPTrack v1.0b2 (Release) 9/29/2013
Fixed crash related to reading properties from mesa
Added frequence setting and track printing to properties
Tracks are now sorted by ID

8/22/2013  ELPTrack v1.0b1 (Release)

Tested under windows 7 with MS VisualStudio Express 2010 installed.  (It should also work without VS but with the MS VS 2010 redistributable pack installed).





