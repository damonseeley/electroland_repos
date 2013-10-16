import SimpleOpenNI.*;

/* --------------------------------------------------------------------------
 * SimpleOpenNI User Test
 * --------------------------------------------------------------------------
 * Processing Wrapper for the OpenNI/Kinect 2 library
 * http://code.google.com/p/simple-openni
 * --------------------------------------------------------------------------
 * prog:  Max Rheiner / Interaction Design / Zhdk / http://iad.zhdk.ch/
 * date:  12/12/2012 (m/d/y)
 * ----------------------------------------------------------------------------
 */

import SimpleOpenNI.*;
import oscP5.*;
import netP5.*;
  
OscP5 oscP5;
NetAddress oscSendToLocation;
OscMessage oscMsg;
int sendPort = 7002;
String sendAddr = "10.22.33.255";

SimpleOpenNI  context;
color[]       userClr = new color[]{ color(255,0,0),
                                     color(0,255,0),
                                     color(0,0,255),
                                     color(255,255,0),
                                     color(255,0,255),
                                     color(0,255,255)
                                   };
PVector com = new PVector();                                   
PVector com2d = new PVector();  

boolean playback = false;
String  recordPath = "1person_userimgtest.oni";

float scale = 2.0f;
PGraphics buffer;

void setup()
{
  frameRate(30);
  size((int)(640*scale),(int)(480*scale));
  
  oscP5 = new OscP5(this,12000);
  oscSendToLocation = new NetAddress(sendAddr,sendPort);
  oscMsg = new OscMessage("/kinectTracks");
  
  if (playback == true) {
    // THIS IS NOT WORKING JUST NOW
    context = new SimpleOpenNI(this,recordPath);
    println("curFramePlayer: " + context.curFramePlayer());
    
  } else {
    context = new SimpleOpenNI(this);
    if(context.isInit() == false)
    {
       println("Can't init SimpleOpenNI, maybe the camera is not connected!"); 
       exit();
       return;  
    }
    // enable depthMap generation 
    context.enableDepth();
     
    // enable skeleton generation for all joints
    context.enableUser();
  }
  
  //buffer = createGraphics(500, 500, JAVA2D);
 
  background(10,0,0);

  stroke(0,0,255);
  strokeWeight(3);
  smooth();  
}



void draw()
{
  // update the cam
  context.update();
  
  // draw depthImageMap
  //image(context.depthImage(),0,0);
  //image(context.userImage(),0,0);
  image(context.userImage(),0,0,(int)(640*scale),(int)(480*scale));
  
  //print(context.getNumberOfUsers());
  //println(context.userTimeStamp());
  
  // draw the skeleton if it's available
  int[] userList = context.getUsers();
  
  for(int i=0;i<userList.length;i++)
  {
    //draw skeletons
    
    /*
    if(context.isTrackingSkeleton(userList[i]))
    {
      stroke(userClr[ (userList[i] - 1) % userClr.length ] );
      drawSkeleton(userList[i]);
    }
    */
    
    // draw the center of mass
    if(context.getCoM(userList[i],com))
    {
      context.convertRealWorldToProjective(com,com2d);
      stroke(100,255,0);
      strokeWeight(1*scale);
      beginShape(LINES);
        vertex(com2d.x*scale,com2d.y*scale - 5);
        vertex(com2d.x*scale,com2d.y*scale + 5);

        vertex(com2d.x*scale - 5,com2d.y*scale);
        vertex(com2d.x*scale + 5,com2d.y*scale);
      endShape();
      
      fill(255,255,255);
      int txtSize = 18;
      textSize(18);
      int offset = 5;
      text("u" + Integer.toString(userList[i]) + "x3d,y3d,z3d: " + (int)com.x + ", " + (int)com.y + ", " + (int)com.z,com2d.x*scale + offset,com2d.y*scale + offset);
      text("x2d,y2d: " + (int)com2d.x + ", " + (int)com2d.y,com2d.x*scale + offset,com2d.y*scale + txtSize + offset);
      //println("u: " + i + " x2d,y2d: " + (int)com2d.x + ", " + (int)com2d.y);
      //String trackParams = i + " " + (int)com.x + " " + (int)com.y + " " + (int)com.z + " " + (int)com2d.x + " " + (int)com2d.y;
      addToOSCByParam(i,(int)com.x,(int)com.y,(int)com.z,(int)com2d.x,(int)com2d.y);
      //addToOSC(trackParams);
    }
  }
  sendOSC();
}

void addToOSC(String str) {
  // send OSC packet of users where params are ID,x3d,y3d,z3d,x2d,y2d
  oscMsg.add(str);
}

void addToOSCByParam(int id, int x3d, int y3d, int z3d, int x2d, int y2d) {
 //only allow messages if users are not at 0,0 in 2D (dead users)
 String trackParams = "";
 String tk = " ";
 if (x2d != 0 || y2d != 0) {
    trackParams = id + tk + x3d + tk + y3d + tk + z3d + tk + x2d + tk + y2d;
    addToOSC(trackParams);
 }

 
}

void sendOSC() {  
  /* send the message */
  oscP5.send(oscMsg, oscSendToLocation);
  
  oscMsg = new OscMessage("/kinectTracks");
}



// -----------------------------------------------------------------
// SimpleOpenNI events

void onNewUser(SimpleOpenNI curContext, int userId)
{
  println("onNewUser - userId: " + userId);
  println("\tstart tracking skeleton");
  
  curContext.startTrackingSkeleton(userId);
}

void onLostUser(SimpleOpenNI curContext, int userId)
{
  println("onLostUser - userId: " + userId);
}

void onVisibleUser(SimpleOpenNI curContext, int userId)
{
  //println("onVisibleUser - userId: " + userId);
}

/* incoming osc message are forwarded to the oscEvent method. */
void oscEvent(OscMessage theOscMessage) {
  /* print the address pattern and the typetag of the received OscMessage */
  print("### received an osc message.");
  print(" addrpattern: "+theOscMessage.addrPattern());
  println(" typetag: "+theOscMessage.typetag());
}


void keyPressed()
{
  switch(key)
  {
  case ' ':
    context.setMirror(!context.mirror());
    break;
  }
}







/*
// draw the skeleton with the selected joints
void drawSkeleton(int userId)
{
  // to get the 3d joint data
  /*
  PVector jointPos = new PVector();
  context.getJointPositionSkeleton(userId,SimpleOpenNI.SKEL_NECK,jointPos);
  println(jointPos);
  */
  /*
  //buffer.beginDraw();
  context.drawLimb(userId, SimpleOpenNI.SKEL_HEAD, SimpleOpenNI.SKEL_NECK);

  context.drawLimb(userId, SimpleOpenNI.SKEL_NECK, SimpleOpenNI.SKEL_LEFT_SHOULDER);
  context.drawLimb(userId, SimpleOpenNI.SKEL_LEFT_SHOULDER, SimpleOpenNI.SKEL_LEFT_ELBOW);
  context.drawLimb(userId, SimpleOpenNI.SKEL_LEFT_ELBOW, SimpleOpenNI.SKEL_LEFT_HAND);

  context.drawLimb(userId, SimpleOpenNI.SKEL_NECK, SimpleOpenNI.SKEL_RIGHT_SHOULDER);
  context.drawLimb(userId, SimpleOpenNI.SKEL_RIGHT_SHOULDER, SimpleOpenNI.SKEL_RIGHT_ELBOW);
  context.drawLimb(userId, SimpleOpenNI.SKEL_RIGHT_ELBOW, SimpleOpenNI.SKEL_RIGHT_HAND);

  context.drawLimb(userId, SimpleOpenNI.SKEL_LEFT_SHOULDER, SimpleOpenNI.SKEL_TORSO);
  context.drawLimb(userId, SimpleOpenNI.SKEL_RIGHT_SHOULDER, SimpleOpenNI.SKEL_TORSO);

  context.drawLimb(userId, SimpleOpenNI.SKEL_TORSO, SimpleOpenNI.SKEL_LEFT_HIP);
  context.drawLimb(userId, SimpleOpenNI.SKEL_LEFT_HIP, SimpleOpenNI.SKEL_LEFT_KNEE);
  context.drawLimb(userId, SimpleOpenNI.SKEL_LEFT_KNEE, SimpleOpenNI.SKEL_LEFT_FOOT);

  context.drawLimb(userId, SimpleOpenNI.SKEL_TORSO, SimpleOpenNI.SKEL_RIGHT_HIP);
  context.drawLimb(userId, SimpleOpenNI.SKEL_RIGHT_HIP, SimpleOpenNI.SKEL_RIGHT_KNEE);
  context.drawLimb(userId, SimpleOpenNI.SKEL_RIGHT_KNEE, SimpleOpenNI.SKEL_RIGHT_FOOT);
  //buffer.endDraw();
}
*/

