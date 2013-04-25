package net.electroland.norfolk.vizapp;
import processing.core.*;
import remixlab.proscene.*;
import saito.objloader.*;
import shapes3d.*;


public class NorfolkVizSketch extends PApplet {
	Scene scene;

	//define a new class for LightObjects
	class LightObject {
	  OBJModel lightModel, participantModel, sensorBeamModel;
	  boolean triggerState;
	  PImage lightTexture;
	  int intensity, lightColor, prevLightColor, fadeOnTime, fadeOffTime;
	  LightObject(boolean passedTriggerState, OBJModel passedLightModel, PImage passedLightTexture, OBJModel passedParticipantModel, OBJModel passedSensorBeamModel, int passedIntensity, int passedLightColor, int passedFadeOnTime, int passedFadeOffTime) {
		triggerState = passedTriggerState;
		lightModel = passedLightModel;
		lightTexture = passedLightTexture;
	    participantModel = passedParticipantModel;
	    sensorBeamModel = passedSensorBeamModel;
	    intensity = passedIntensity;
	    lightColor = passedLightColor;
	    fadeOnTime = passedFadeOnTime;
	    fadeOffTime = passedFadeOffTime;
	  }
	}

	//declare display flags
	boolean showParticipants = true;
	boolean showSensorBeams = true;
	boolean showEnvironment = true;

	//declare all OBJModels
	OBJModel sculpture, environment, modelpB01, modelpB02, modelpB03, modelpF01, modelpF02, modelpF03, modelpF05, modelpF06, modelpF08, modelpF09, modelpF10, modelpF11, modelpF12, modelsB01, modelsB02, modelsB03, modelsF01, modelsF02, modelsF03, modelsF05, modelsF06, modelsF08, modelsF09, modelsF10, modelsF11, modelsF12, modelsT01;

	//declare all lightVolumes
	OBJModel volumeB01, volumeB02, volumeB03, volumeC01A, volumeC01B, volumeC02A, volumeC02B, volumeC03A, volumeC03B, volumeF01, volumeF02, volumeF03, volumeF05, volumeF06, volumeF08, volumeF09, volumeF10, volumeF11, volumeF12, volumeL01, volumeL02, volumeV01, volumeV02, volumeV03, volumeV04;
	
	//declare all texture PImages
	PImage texB01, texB02, texB03, texC01A, texC01B, texC02A, texC02B, texC03A, texC03B, texF01, texF02, texF03, texF05, texF06, texF08, texF09, texF10, texF11, texF12, texL01, texL02, texV01, texV02, texV03, texV04;
	
	//Declare an array to hold all the LightObjects
	LightObject[] allLightObjects = new LightObject[25];

	public void setup()
	{
	  
	  //Scene instantiation
	  size(800, 600, P3D);
	  scene = new Scene(this);
	  scene.setGridIsDrawn(false);
	  scene.setAxisIsDrawn(false);
	  scene.setRadius(600);
	  
	  //Register a CAD Camera profile and name it "CAD_CAM"
	  scene.registerCameraProfile(new CameraProfile(scene, "CAD_CAM", CameraProfile.Mode.CAD));
	  //Set the CAD_CAM as the current camera profile
	  scene.setCurrentCameraProfile("CAD_CAM");
	  //Unregister the  first-person camera profile (i.e., leave WHEELED_ARCBALL and CAD_CAM)
	  scene.unregisterCameraProfile("FIRST_PERSON");
	  //Set right handed world frame (useful for architects...)
	  scene.setRightHanded();
	  scene.camera().frame().setCADAxis(new PVector(0, 1, 0));
	  scene.camera().frame().setRotationSensitivity(1);
	  scene.camera().frame().setSpinningFriction(1);
	  scene.camera().frame().setTossingFriction(1);
	  
	  //Define all OBJModels and their geo  
	  sculpture = new OBJModel(this, "../depends/models/sculpture.obj", "absolute", TRIANGLES);
	  environment = new OBJModel(this, "../depends/models/environment.obj", "absolute", TRIANGLES);
	  volumeB01 = new OBJModel(this, "../depends/models/B01.obj", "absolute", TRIANGLES);
	  volumeB02 = new OBJModel(this, "../depends/models/B02.obj", "absolute", TRIANGLES);
	  volumeB03 = new OBJModel(this, "../depends/models/B03.obj", "absolute", TRIANGLES);
	  volumeC01A = new OBJModel(this, "../depends/models/C01A.obj", "absolute", TRIANGLES);
	  volumeC01B = new OBJModel(this, "../depends/models/C01B.obj", "absolute", TRIANGLES);
	  volumeC02A = new OBJModel(this, "../depends/models/C02A.obj", "absolute", TRIANGLES);
	  volumeC02B = new OBJModel(this, "../depends/models/C02B.obj", "absolute", TRIANGLES);
	  volumeC03A = new OBJModel(this, "../depends/models/C03A.obj", "absolute", TRIANGLES);
	  volumeC03B = new OBJModel(this, "../depends/models/C03B.obj", "absolute", TRIANGLES);
	  volumeF01 = new OBJModel(this, "../depends/models/F01.obj", "absolute", TRIANGLES);
	  volumeF02 = new OBJModel(this, "../depends/models/F02.obj", "absolute", TRIANGLES);
	  volumeF03 = new OBJModel(this, "../depends/models/F03.obj", "absolute", TRIANGLES);
	  volumeF05 = new OBJModel(this, "../depends/models/F05.obj", "absolute", TRIANGLES);
	  volumeF06 = new OBJModel(this, "../depends/models/F06.obj", "absolute", TRIANGLES);
	  volumeF08 = new OBJModel(this, "../depends/models/F08.obj", "absolute", TRIANGLES);
	  volumeF09 = new OBJModel(this, "../depends/models/F09.obj", "absolute", TRIANGLES);
	  volumeF10 = new OBJModel(this, "../depends/models/F10.obj", "absolute", TRIANGLES);
	  volumeF11 = new OBJModel(this, "../depends/models/F11.obj", "absolute", TRIANGLES);
	  volumeF12 = new OBJModel(this, "../depends/models/F12.obj", "absolute", TRIANGLES);
	  volumeL01 = new OBJModel(this, "../depends/models/L01.obj", "absolute", TRIANGLES);
	  volumeL02 = new OBJModel(this, "../depends/models/L02.obj", "absolute", TRIANGLES);
	  volumeV01 = new OBJModel(this, "../depends/models/V01.obj", "absolute", TRIANGLES);
	  volumeV02 = new OBJModel(this, "../depends/models/V02.obj", "absolute", TRIANGLES);
	  volumeV03 = new OBJModel(this, "../depends/models/V03.obj", "absolute", TRIANGLES);
	  volumeV04 = new OBJModel(this, "../depends/models/V04.obj", "absolute", TRIANGLES);
	  modelpB01 = new OBJModel(this, "../depends/models/pB01.obj", "absolute", TRIANGLES);
	  modelpB02 = new OBJModel(this, "../depends/models/pB02.obj", "absolute", TRIANGLES);
	  modelpB03 = new OBJModel(this, "../depends/models/pB03.obj", "absolute", TRIANGLES);
	  modelpF01 = new OBJModel(this, "../depends/models/pF01.obj", "absolute", TRIANGLES);
	  modelpF02 = new OBJModel(this, "../depends/models/pF02.obj", "absolute", TRIANGLES);
	  modelpF03 = new OBJModel(this, "../depends/models/pF03.obj", "absolute", TRIANGLES);
	  modelpF05 = new OBJModel(this, "../depends/models/pF05.obj", "absolute", TRIANGLES);
	  modelpF06 = new OBJModel(this, "../depends/models/pF06.obj", "absolute", TRIANGLES);
	  modelpF08 = new OBJModel(this, "../depends/models/pF08.obj", "absolute", TRIANGLES);
	  modelpF09 = new OBJModel(this, "../depends/models/pF09.obj", "absolute", TRIANGLES);
	  modelpF10 = new OBJModel(this, "../depends/models/pF10.obj", "absolute", TRIANGLES);
	  modelpF11 = new OBJModel(this, "../depends/models/pF11.obj", "absolute", TRIANGLES);
	  modelpF12 = new OBJModel(this, "../depends/models/pF12.obj", "absolute", TRIANGLES);
	  modelsB01 = new OBJModel(this, "../depends/models/sB01.obj", "absolute", TRIANGLES);
	  modelsB02 = new OBJModel(this, "../depends/models/sB02.obj", "absolute", TRIANGLES);
	  modelsB03 = new OBJModel(this, "../depends/models/sB03.obj", "absolute", TRIANGLES);
	  modelsF01 = new OBJModel(this, "../depends/models/sF01.obj", "absolute", TRIANGLES);
	  modelsF02 = new OBJModel(this, "../depends/models/sF02.obj", "absolute", TRIANGLES);
	  modelsF03 = new OBJModel(this, "../depends/models/sF03.obj", "absolute", TRIANGLES);
	  modelsF05 = new OBJModel(this, "../depends/models/sF05.obj", "absolute", TRIANGLES);
	  modelsF06 = new OBJModel(this, "../depends/models/sF06.obj", "absolute", TRIANGLES);
	  modelsF08 = new OBJModel(this, "../depends/models/sF08.obj", "absolute", TRIANGLES);
	  modelsF09 = new OBJModel(this, "../depends/models/sF09.obj", "absolute", TRIANGLES);
	  modelsF10 = new OBJModel(this, "../depends/models/sF10.obj", "absolute", TRIANGLES);
	  modelsF11 = new OBJModel(this, "../depends/models/sF11.obj", "absolute", TRIANGLES);
	  modelsF12 = new OBJModel(this, "../depends/models/sF12.obj", "absolute", TRIANGLES);
	  
	  //Define textures
	  texB01 = loadImage("../depends/models/lightGrad.jpg");
	  texB02 = loadImage("../depends/models/lightGrad.jpg");
	  texB03 = loadImage("../depends/models/lightGrad.jpg");
	  texC01A = loadImage("../depends/models/lightGrad.jpg");
	  texC01B = loadImage("../depends/models/lightGrad.jpg");
	  texC02A = loadImage("../depends/models/lightGrad.jpg");
	  texC02B = loadImage("../depends/models/lightGrad.jpg");
	  texC03A = loadImage("../depends/models/lightGrad.jpg");
	  texC03B = loadImage("../depends/models/lightGrad.jpg");
	  texF01 = loadImage("../depends/models/lightGrad.jpg");
	  texF02 = loadImage("../depends/models/lightGrad.jpg");
	  texF03 = loadImage("../depends/models/lightGrad.jpg");
	  texF05 = loadImage("../depends/models/lightGrad.jpg");
	  texF06 = loadImage("../depends/models/lightGrad.jpg");
	  texF08 = loadImage("../depends/models/lightGrad.jpg");
	  texF09 = loadImage("../depends/models/lightGrad.jpg");
	  texF10 = loadImage("../depends/models/lightGrad.jpg");
	  texF11 = loadImage("../depends/models/lightGrad.jpg");
	  texF12 = loadImage("../depends/models/lightGrad.jpg");
	  texL01 = loadImage("../depends/models/lightGrad.jpg");
	  texL02 = loadImage("../depends/models/lightGrad.jpg");
	  texV01 = loadImage("../depends/models/lightGrad.jpg");
	  texV02 = loadImage("../depends/models/lightGrad.jpg");
	  texV03 = loadImage("../depends/models/lightGrad.jpg");
	  texV04 = loadImage("../depends/models/lightGrad.jpg");
	  
	  
	  //initialize LightObjects
	  LightObject B01 = new LightObject(false, volumeB01, texB01, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject B02 = new LightObject(false, volumeB02, texB02, modelpB02, modelsB02, 255, color(255, 255, 255), 2000, 2000);
	  LightObject B03 = new LightObject(false, volumeB03, texB03, modelpB03, modelsB03, 255, color(255, 255, 255), 2000, 2000);
	  LightObject C01A = new LightObject(false, volumeC01A, texC01A, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject C01B = new LightObject(false, volumeC01B, texC01B, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject C02A = new LightObject(false, volumeC02A, texC02A, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject C02B = new LightObject(false, volumeC02B, texC02B, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject C03A = new LightObject(false, volumeC03A, texC03A, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject C03B = new LightObject(false, volumeC03B, texC03B, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject F01 = new LightObject(false, volumeF01, texF01, modelpF01, modelsF01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject F02 = new LightObject(false, volumeF02, texF02, modelpF02, modelsF02, 255, color(255, 255, 255), 2000, 2000);
	  LightObject F03 = new LightObject(false, volumeF03, texF03, modelpF03, modelsF03, 255, color(255, 255, 255), 2000, 2000);
	  LightObject F05 = new LightObject(false, volumeF05, texF05, modelpF05, modelsF05, 255, color(255, 255, 255), 2000, 2000);
	  LightObject F06 = new LightObject(false, volumeF06, texF06, modelpF06, modelsF06, 255, color(255, 255, 255), 2000, 2000);
	  LightObject F08 = new LightObject(false, volumeF08, texF08, modelpF08, modelsF08, 255, color(255, 255, 255), 2000, 2000);
	  LightObject F09 = new LightObject(false, volumeF09, texF09, modelpF09, modelsF09, 255, color(255, 255, 255), 2000, 2000);
	  LightObject F10 = new LightObject(false, volumeF10, texF10, modelpF10, modelsF10, 255, color(255, 255, 255), 2000, 2000);
	  LightObject F11 = new LightObject(false, volumeF11, texF11, modelpF11, modelsF11, 255, color(255, 255, 255), 2000, 2000);
	  LightObject F12 = new LightObject(false, volumeF12, texF12, modelpF12, modelsF12, 255, color(255, 255, 255), 2000, 2000);
	  LightObject L01 = new LightObject(false, volumeL01, texL01, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject L02 = new LightObject(false, volumeL02, texL02, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject V01 = new LightObject(false, volumeV01, texV01, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject V02 = new LightObject(false, volumeV02, texV02, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject V03 = new LightObject(false, volumeV03, texV03, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  LightObject V04 = new LightObject(false, volumeV04, texV04, modelpB01, modelsB01, 255, color(255, 255, 255), 2000, 2000);
	  

	  //build the array of all light objects, and initialize them
	  allLightObjects[0] = B01;
	  allLightObjects[1] = B02;
	  allLightObjects[2] = B03;
	  allLightObjects[3] = C01A;
	  allLightObjects[4] = C01B;
	  allLightObjects[5] = C02A;
	  allLightObjects[6] = C02B;
	  allLightObjects[7] = C03A;
	  allLightObjects[8] = C03B;
	  allLightObjects[9] = F01;
	  allLightObjects[10] = F02;
	  allLightObjects[11] = F03;
	  allLightObjects[12] = F05;
	  allLightObjects[13] = F06;
	  allLightObjects[14] = F08;
	  allLightObjects[15] = F09;
	  allLightObjects[16] = F10;
	  allLightObjects[17] = F11;
	  allLightObjects[18] = F12;
	  allLightObjects[19] = L01;
	  allLightObjects[20] = L02;
	  allLightObjects[21] = V01;
	  allLightObjects[22] = V02;
	  allLightObjects[23] = V03;
	  allLightObjects[24] = V04;
	  

	  //Set stroke color to white, then hide strokes
	  stroke(255);
	  noStroke();
	}



	public void draw()
	{
	    background(129);
	    lights();
	    
	    blendMode(NORMAL);
	    sculpture.draw();
	    
	    if (showEnvironment == true) environment.draw();
	    //iterate through allLightObjects and draw
	    for (int i = 0; i < allLightObjects.length; i = i+1) {
	    	blendMode(NORMAL);
	    	if (showParticipants == true) allLightObjects[i].participantModel.draw();
	    	if (showSensorBeams == true) allLightObjects[i].sensorBeamModel.draw();
	    	//blendMode(SCREEN);
	        if (allLightObjects[i].intensity > 0) {
	        	allLightObjects[i].lightModel.setTexture(texV04);
	        	allLightObjects[i].lightModel.enableMaterial();
	        	allLightObjects[i].lightModel.enableTexture();
	        	allLightObjects[i].lightModel.draw();
	      }
	      
	    }
	    

	}

	boolean bTexture = true;
	boolean bStroke = false;
}