package net.electroland.norfolk.vizapp;
import processing.core.*;
import remixlab.proscene.*;
//import saito.objloader.*;


public class norfolkVizSketch extends PApplet {
	Scene scene;

	//define a new class for lightObjects
	class lightObject {
	  PShape lightModel, participantModel, sensorBeamModel;
	  int intensity;
	  int lightColor, prevLightColor;
	  lightObject(PShape tempLightModel, PShape tempParticipantModel, PShape tempSensorBeamModel, int tempIntensity, int tempLightColor, int tempPrevLightColor) {
	    lightModel = tempLightModel;
	    participantModel = tempParticipantModel;
	    sensorBeamModel = tempSensorBeamModel;
	    intensity = tempIntensity;
	    lightColor = tempLightColor;
	    prevLightColor = tempPrevLightColor;
	  }
	}

	//declare display flags globally
	boolean showParticipants = true;
	boolean showSensorBeams = true;
	boolean showEnvironment = true;

	//decalre all PShapes globally
	PShape sculpture, environment, modelB01, modelB02, modelB03, modelC01A, modelC01B, modelC02A, modelC02B, modelC03A, modelC03B, modelF01, modelF02, modelF03, modelF05, modelF06, modelF08, modelF09, modelF10, modelF11, modelF12, modelL01, modelL02, modelpB01, modelpB02, modelpB03, modelpF01, modelpF02, modelpF03, modelpF05, modelpF06, modelpF08, modelpF09, modelpF10, modelpF11, modelpF12, modelsB01, modelsB02, modelsB03, modelsF01, modelsF02, modelsF03, modelsF05, modelsF06, modelsF08, modelsF09, modelsF10, modelsF11, modelsF12, modelsT01, modelV01, modelV02, modelV03, modelV04;

	//Testing - let's try to build an array and draw with it
	PShape[] allModels = new PShape[51];

	//Declare an array to hold all the lightObjects
	lightObject[] allLightObjects = new lightObject[25];

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
	  
	  //Define all objects and their geo  
	  sculpture = loadShape("sculpture.obj");
	  environment = loadShape("environment.obj");
	  modelB01 = loadShape("B01.obj");
	  modelB02 = loadShape("B02.obj");
	  modelB03 = loadShape("B03.obj");
	  modelC01A = loadShape("C01A.obj");
	  modelC01B = loadShape("C01B.obj");
	  modelC02A = loadShape("C02A.obj");
	  modelC02B = loadShape("C02B.obj");
	  modelC03A = loadShape("C03A.obj");
	  modelC03B = loadShape("C03B.obj");
	  modelF01 = loadShape("F01.obj");
	  modelF02 = loadShape("F02.obj");
	  modelF03 = loadShape("F03.obj");
	  modelF05 = loadShape("F05.obj");
	  modelF06 = loadShape("F06.obj");
	  modelF08 = loadShape("F08.obj");
	  modelF09 = loadShape("F09.obj");
	  modelF10 = loadShape("F10.obj");
	  modelF11 = loadShape("F11.obj");
	  modelF12 = loadShape("F12.obj");
	  modelL01 = loadShape("L01.obj");
	  modelL02 = loadShape("L02.obj");
	  modelpB01 = loadShape("pB01.obj");
	  modelpB02 = loadShape("pB02.obj");
	  modelpB03 = loadShape("pB03.obj");
	  modelpF01 = loadShape("pF01.obj");
	  modelpF02 = loadShape("pF02.obj");
	  modelpF03 = loadShape("pF03.obj");
	  modelpF05 = loadShape("pF05.obj");
	  modelpF06 = loadShape("pF06.obj");
	  modelpF08 = loadShape("pF08.obj");
	  modelpF09 = loadShape("pF09.obj");
	  modelpF10 = loadShape("pF10.obj");
	  modelpF11 = loadShape("pF11.obj");
	  modelpF12 = loadShape("pF12.obj");
	  modelsB01 = loadShape("sB01.obj");
	  modelsB02 = loadShape("sB02.obj");
	  modelsB03 = loadShape("sB03.obj");
	  modelsF01 = loadShape("sF01.obj");
	  modelsF02 = loadShape("sF02.obj");
	  modelsF03 = loadShape("sF03.obj");
	  modelsF05 = loadShape("sF05.obj");
	  modelsF06 = loadShape("sF06.obj");
	  modelsF08 = loadShape("sF08.obj");
	  modelsF09 = loadShape("sF09.obj");
	  modelsF10 = loadShape("sF10.obj");
	  modelsF11 = loadShape("sF11.obj");
	  modelsF12 = loadShape("sF12.obj");
	  modelV01 = loadShape("V01.obj");
	  modelV02 = loadShape("V02.obj");
	  modelV03 = loadShape("V03.obj");
	  modelV04 = loadShape("V04.obj");

	  //initialize lightObjects
	  lightObject B01 = new lightObject(modelB01, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject B02 = new lightObject(modelB02, modelpB02, modelsB02, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject B03 = new lightObject(modelB03, modelpB03, modelsB03, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject C01A = new lightObject(modelC01A, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject C01B = new lightObject(modelC01B, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject C02A = new lightObject(modelC02A, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject C02B = new lightObject(modelC02B, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject C03A = new lightObject(modelC03A, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject C03B = new lightObject(modelC03B, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject F01 = new lightObject(modelF01, modelpF01, modelsF01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject F02 = new lightObject(modelF02, modelpF02, modelsF02, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject F03 = new lightObject(modelF03, modelpF03, modelsF03, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject F05 = new lightObject(modelF05, modelpF05, modelsF05, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject F06 = new lightObject(modelF06, modelpF06, modelsF06, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject F08 = new lightObject(modelF08, modelpF08, modelsF08, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject F09 = new lightObject(modelF09, modelpF09, modelsF09, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject F10 = new lightObject(modelF10, modelpF10, modelsF10, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject F11 = new lightObject(modelF11, modelpF11, modelsF11, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject F12 = new lightObject(modelF12, modelpF12, modelsF12, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject L01 = new lightObject(modelL01, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject L02 = new lightObject(modelL02, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject V01 = new lightObject(modelV01, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject V02 = new lightObject(modelV02, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject V03 = new lightObject(modelV03, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  lightObject V04 = new lightObject(modelV04, modelpB01, modelsB01, 255, color(255, 255, 255), color(255, 255, 255));
	  
	  //fill that test array
	  allModels[0] = sculpture;
	  allModels[1] = environment;

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
	    //allModels[0].draw();
	    
	    shape(sculpture);
	    
	    if (showEnvironment == true) shape(environment);
	    
	    /*//iterate through allLightObjects and draw
	    for (int i = 0; i < allLightObjects.length; i = i+1) {
	      if (showParticipants == true) allLightObjects[i].participantModel.draw();
	      if (showSensorBeams == true) allLightObjects[i].sensorBeamModel.draw();
	      if (allLightObjects[i].intensity > 0) {
	        allLightObjects[i].lightModel.draw();
	      }*/
	      
	    }
	    

	//}

	boolean bTexture = true;
	boolean bStroke = false;
}
