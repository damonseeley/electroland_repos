package net.electroland.norfolk.vizapp;
import processing.core.*;
import remixlab.proscene.*;
import saito.objloader.*;


public class NorfolkVizSketch extends PApplet {
	Scene scene;

	//define a new class for lightObjects
	class lightObject {
	  OBJModel lightModel, participantModel, sensorBeamModel;
	  int intensity;
	  int lightColor, prevLightColor;
	  lightObject(OBJModel tempLightModel, OBJModel tempParticipantModel, OBJModel tempSensorBeamModel, int tempIntensity, int tempLightColor, int tempPrevLightColor) {
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

	//declare all OBJModels globally
	OBJModel sculpture, environment, modelB01, modelB02, modelB03, modelC01A, modelC01B, modelC02A, modelC02B, modelC03A, modelC03B, modelF01, modelF02, modelF03, modelF05, modelF06, modelF08, modelF09, modelF10, modelF11, modelF12, modelL01, modelL02, modelpB01, modelpB02, modelpB03, modelpF01, modelpF02, modelpF03, modelpF05, modelpF06, modelpF08, modelpF09, modelpF10, modelpF11, modelpF12, modelsB01, modelsB02, modelsB03, modelsF01, modelsF02, modelsF03, modelsF05, modelsF06, modelsF08, modelsF09, modelsF10, modelsF11, modelsF12, modelsT01, modelV01, modelV02, modelV03, modelV04;

	//Testing - let's try to build an array and draw with it
	OBJModel[] allModels = new OBJModel[51];

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
	  sculpture = new OBJModel(this, "../depends/models/sculpture.obj", "absolute", TRIANGLES);
	  environment = new OBJModel(this, "../depends/models/environment.obj", "absolute", TRIANGLES);
	  modelB01 = new OBJModel(this, "../depends/models/B01.obj", "absolute", TRIANGLES);
	  modelB02 = new OBJModel(this, "../depends/models/B02.obj", "absolute", TRIANGLES);
	  modelB03 = new OBJModel(this, "../depends/models/B03.obj", "absolute", TRIANGLES);
	  modelC01A = new OBJModel(this, "../depends/models/C01A.obj", "absolute", TRIANGLES);
	  modelC01B = new OBJModel(this, "../depends/models/C01B.obj", "absolute", TRIANGLES);
	  modelC02A = new OBJModel(this, "../depends/models/C02A.obj", "absolute", TRIANGLES);
	  modelC02B = new OBJModel(this, "../depends/models/C02B.obj", "absolute", TRIANGLES);
	  modelC03A = new OBJModel(this, "../depends/models/C03A.obj", "absolute", TRIANGLES);
	  modelC03B = new OBJModel(this, "../depends/models/C03B.obj", "absolute", TRIANGLES);
	  modelF01 = new OBJModel(this, "../depends/models/F01.obj", "absolute", TRIANGLES);
	  modelF02 = new OBJModel(this, "../depends/models/F02.obj", "absolute", TRIANGLES);
	  modelF03 = new OBJModel(this, "../depends/models/F03.obj", "absolute", TRIANGLES);
	  modelF05 = new OBJModel(this, "../depends/models/F05.obj", "absolute", TRIANGLES);
	  modelF06 = new OBJModel(this, "../depends/models/F06.obj", "absolute", TRIANGLES);
	  modelF08 = new OBJModel(this, "../depends/models/F08.obj", "absolute", TRIANGLES);
	  modelF09 = new OBJModel(this, "../depends/models/F09.obj", "absolute", TRIANGLES);
	  modelF10 = new OBJModel(this, "../depends/models/F10.obj", "absolute", TRIANGLES);
	  modelF11 = new OBJModel(this, "../depends/models/F11.obj", "absolute", TRIANGLES);
	  modelF12 = new OBJModel(this, "../depends/models/F12.obj", "absolute", TRIANGLES);
	  modelL01 = new OBJModel(this, "../depends/models/L01.obj", "absolute", TRIANGLES);
	  modelL02 = new OBJModel(this, "../depends/models/L02.obj", "absolute", TRIANGLES);
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
	  modelV01 = new OBJModel(this, "../depends/models/V01.obj", "absolute", TRIANGLES);
	  modelV02 = new OBJModel(this, "../depends/models/V02.obj", "absolute", TRIANGLES);
	  modelV03 = new OBJModel(this, "../depends/models/V03.obj", "absolute", TRIANGLES);
	  modelV04 = new OBJModel(this, "../depends/models/V04.obj", "absolute", TRIANGLES);

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
	    
	    sculpture.draw();
	    
	    if (showEnvironment == true) environment.draw();
	    
	    //iterate through allLightObjects and draw
	    for (int i = 0; i < allLightObjects.length; i = i+1) {
	      if (showParticipants == true) allLightObjects[i].participantModel.draw();
	      if (showSensorBeams == true) allLightObjects[i].sensorBeamModel.draw();
	      if (allLightObjects[i].intensity > 0) {
	        allLightObjects[i].lightModel.draw();
	      }
	      
	    }
	    

	}

	boolean bTexture = true;
	boolean bStroke = false;
}