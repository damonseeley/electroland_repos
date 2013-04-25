package net.electroland.norfolk.vizapp;
import java.awt.Color;
import processing.core.*;
import remixlab.proscene.*;
import saito.objloader.*;
import java.util.Random;


public class NorfolkVizSketch extends PApplet {
	Scene scene;

	//define a new class for LightObjects
	//
	class LightObject {
		OBJModel lightModel, participantModel, sensorBeamModel;
		boolean triggerState;
		int lightColor;
		PImage lightTexture, lightMask;
		double lightIntensity;
		LightObject(OBJModel passedLightModel, OBJModel passedParticipantModel, OBJModel passedSensorBeamModel) {
			triggerState = false;
			lightModel = passedLightModel;
			participantModel = passedParticipantModel;
			sensorBeamModel = passedSensorBeamModel;
			lightTexture = createImage(128, 128, RGB);
			lightMask = loadImage("../depends/models/lightGrad.jpg");
			lightIntensity = 0.5;
			lightColor = color(255,255,0);
			lightTexture.loadPixels();
			for (int i = 0; i < lightTexture.pixels.length; i++) {
				lightTexture.pixels[i] = lightColor; 
			}
			lightTexture.mask(lightMask);
			lightTexture.updatePixels();
			lightModel.setTexture(lightTexture);
			lightModel.enableTexture();
		}
	  
	  //setLightColor makes a new PImage texture for each light
	  //Intensity will usually be 1.0, but exists in case we want that functionality later 
		public void setLightColor(String nameOfLight, Color lightColor) {
			int r = Math.min(255, (int) (lightColor.getRed() * lightIntensity));
			int g = Math.min(255, (int) (lightColor.getGreen() * lightIntensity));
			int b = Math.min(255, (int) (lightColor.getBlue() * lightIntensity));
			lightTexture.loadPixels();
			for (int i = 0; i < lightTexture.pixels.length; i++) {
				lightTexture.pixels[i] = color(r,g,b); 
			}
			lightTexture.mask(lightMask);
			lightTexture.updatePixels();		  
		}
	  
		public void setSensorState(String sensorName, Boolean isOn) {
			triggerState = isOn;
			System.err.println("Trigger " + sensorName + "is tripped high: " + triggerState);

		}

	}

	//declare and set some display flags
	boolean showParticipants = true;
	boolean showSensorBeams = true;
	boolean showEnvironment = true;
	boolean bTexture = true;
	boolean bStroke = false;
	int zoomLevel = 1200;

	//declare all OBJModels
	OBJModel sculpture, environment, volumeB01, volumeB02, volumeB03, volumeC01A, volumeC01B, volumeC02A, volumeC02B, volumeC03A, volumeC03B, volumeF01, volumeF02, volumeF03, volumeF05, volumeF06, volumeF08, volumeF09, volumeF10, volumeF11, volumeF12, volumeL01, volumeL02, volumeV01, volumeV02, volumeV03, volumeV04, modelpB01, modelpB02, modelpB03, modelpF01, modelpF02, modelpF03, modelpF05, modelpF06, modelpF08, modelpF09, modelpF10, modelpF11, modelpF12, modelsB01, modelsB02, modelsB03, modelsF01, modelsF02, modelsF03, modelsF05, modelsF06, modelsF08, modelsF09, modelsF10, modelsF11, modelsF12, modelsT01;

	
	//Declare an array to hold all the LightObjects
	//This will be a hashmap soon
	LightObject[] allLightObjects = new LightObject[25];

	public void setup()
	{
	  //Scene/proscene instantiation
	  size(800, 600, P3D);
	  scene = new Scene(this);
	  scene.setGridIsDrawn(false);
	  scene.setAxisIsDrawn(false);
	  scene.setRadius(2000);
	  scene.camera().setPosition(new PVector(0, 50, zoomLevel));
	  scene.disableKeyboardHandling();
	  
	  
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
	  
	  //initialize LightObjects
	  LightObject B01 = new LightObject(volumeB01, modelpB01, modelsB01);
	  LightObject B02 = new LightObject(volumeB02, modelpB02, modelsB02);
	  LightObject B03 = new LightObject(volumeB03, modelpB03, modelsB03);
	  LightObject C01A = new LightObject(volumeC01A, modelpB01, modelsB01);
	  LightObject C01B = new LightObject(volumeC01B, modelpB01, modelsB01);
	  LightObject C02A = new LightObject(volumeC02A, modelpB01, modelsB01);
	  LightObject C02B = new LightObject(volumeC02B, modelpB01, modelsB01);
	  LightObject C03A = new LightObject(volumeC03A, modelpB01, modelsB01);
	  LightObject C03B = new LightObject(volumeC03B, modelpB01, modelsB01);
	  LightObject F01 = new LightObject(volumeF01, modelpF01, modelsF01);
	  LightObject F02 = new LightObject(volumeF02, modelpF02, modelsF02);
	  LightObject F03 = new LightObject(volumeF03, modelpF03, modelsF03);
	  LightObject F05 = new LightObject(volumeF05, modelpF05, modelsF05);
	  LightObject F06 = new LightObject(volumeF06, modelpF06, modelsF06);
	  LightObject F08 = new LightObject(volumeF08, modelpF08, modelsF08);
	  LightObject F09 = new LightObject(volumeF09, modelpF09, modelsF09);
	  LightObject F10 = new LightObject(volumeF10, modelpF10, modelsF10);
	  LightObject F11 = new LightObject(volumeF11, modelpF11, modelsF11);
	  LightObject F12 = new LightObject(volumeF12, modelpF12, modelsF12);
	  LightObject L01 = new LightObject(volumeL01, modelpB01, modelsB01);
	  LightObject L02 = new LightObject(volumeL02, modelpB01, modelsB01);
	  LightObject V01 = new LightObject(volumeV01, modelpB01, modelsB01);
	  LightObject V02 = new LightObject(volumeV02, modelpB01, modelsB01);
	  LightObject V03 = new LightObject(volumeV03, modelpB01, modelsB01);
	  LightObject V04 = new LightObject(volumeV04, modelpB01, modelsB01);
	  

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

		blendMode(BLEND);
		sculpture.draw();

		if (showEnvironment == true) environment.draw();
		//iterate through allLightObjects and draw
		for (int i = 0; i < allLightObjects.length; i = i+1) {
			if (showParticipants == true) {
				if (allLightObjects[i].triggerState == true) allLightObjects[i].participantModel.draw();
			}
			if (showSensorBeams == true) {
				if (allLightObjects[i].triggerState == true) allLightObjects[i].sensorBeamModel.draw();
			}
		}
		blendMode(ADD);
		for (int i = 0; i < allLightObjects.length; i = i+1) {
			allLightObjects[i].lightModel.draw();
		}

	}
	
	public void keyPressed()
	{
		if(key == ',') {
			zoomLevel += 100;
			scene.camera().setPosition(new PVector(0, 50, zoomLevel));
		}
		
		if(key == '.') {
			zoomLevel -= 100;
			scene.camera().setPosition(new PVector(0, 50, zoomLevel));
		}
		
		if(key == 'r') {
			Random rand = new Random();
			for (int i = 0; i < allLightObjects.length; i = i+1) {
				float rRand = rand.nextInt(255);
				float gRand = rand.nextInt(255);
				float bRand = rand.nextInt(255);
				allLightObjects[i].lightTexture.loadPixels();
				for (int ii = 0; ii < allLightObjects[i].lightTexture.pixels.length; ii++) {
					allLightObjects[i].lightTexture.pixels[ii] = color(rRand,gRand,bRand); 
				}
				allLightObjects[i].lightTexture.mask(allLightObjects[i].lightMask);
				allLightObjects[i].lightTexture.updatePixels();
			}
		}
			
		if(key == 'p') {
			if(!showParticipants) {
	            showParticipants = true;
	        } 
	        else {
	        	showParticipants = false;
	        }

		}
		
		if(key == 'b') {
			if(!showSensorBeams) {
				showSensorBeams = true;
	        } 
	        else {
	        	showSensorBeams = false;
	        }

		}
		
		if(key == 'e') {
			if(!showEnvironment) {
				showEnvironment = true;
	        } 
	        else {
	        	showEnvironment = false;
	        }

		}
			
		if(key == '1') {
			if(!allLightObjects[1].triggerState) {
				allLightObjects[1].triggerState = true;
	        } 
	        else {
	        	allLightObjects[1].triggerState = false;
	        }

		}
		if(key == '2') {
			if(!allLightObjects[2].triggerState) {
				allLightObjects[2].triggerState = true;
	        } 
	        else {
	        	allLightObjects[2].triggerState = false;
	        }

		}
			
		if(key == '3') {
			if(!allLightObjects[9].triggerState) {
				allLightObjects[9].triggerState = true;
	        } 
	        else {
	        	allLightObjects[9].triggerState = false;
	        }

		}
			
		if(key == '4') {
			if(!allLightObjects[10].triggerState) {
				allLightObjects[10].triggerState = true;
	        } 
	        else {
	        	allLightObjects[10].triggerState = false;
	        }

		}
		
		if(key == '5') {
			if(!allLightObjects[11].triggerState) {
				allLightObjects[11].triggerState = true;
	        } 
	        else {
	        	allLightObjects[11].triggerState = false;
	        }

		}
			

			
	}
}