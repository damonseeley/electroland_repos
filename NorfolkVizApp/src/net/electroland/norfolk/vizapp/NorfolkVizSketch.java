package net.electroland.norfolk.vizapp;
import java.awt.Color;
import java.util.HashMap;
import java.util.Random;

import net.electroland.norfolk.core.viz.VizOSCListener;
import net.electroland.norfolk.core.viz.VizOSCReceiver;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.ShutdownThread;
import net.electroland.utils.Shutdownable;
import processing.core.PApplet;
import processing.core.PImage;
import processing.core.PVector;
import remixlab.proscene.CameraProfile;
import remixlab.proscene.Scene;
import saito.objloader.OBJModel;


public class NorfolkVizSketch extends PApplet implements VizOSCListener, Shutdownable {
	/**
     * 
     */
    private static final long serialVersionUID = 399552159162053681L;
    Scene scene;

	//define a new class for LightObjects
	//
	class LightObject {
		OBJModel lightModel, participantModel, sensorBeamModel, lightPoolModel;
		boolean triggerState;
		int lightColor;
		PImage lightTexture, lightMask, lightPoolTexture, lightPoolMask;
		double lightIntensity;
		LightObject(OBJModel passedLightModel, OBJModel passedParticipantModel, OBJModel passedSensorBeamModel, OBJModel passedLightPoolModel, double passedLightIntensity) {
			triggerState = false;
			lightModel = passedLightModel;
			lightPoolModel = passedLightPoolModel;
			participantModel = passedParticipantModel;
			sensorBeamModel = passedSensorBeamModel;
			sensorBeamModel.setTexture(beamTexture);
			sensorBeamModel.enableTexture();
			lightTexture = createImage(128, 128, RGB);
			lightPoolTexture = createImage(128, 128, RGB);
			lightMask = loadImage("../depends/models/lightGrad.jpg");
			lightPoolMask = loadImage("../depends/models/lightPoolMask.jpg");
			lightIntensity = passedLightIntensity;
			lightColor = color(255,255,0);
			
			//fill the light texture with color
			lightTexture.loadPixels();
			for (int i = 0; i < lightTexture.pixels.length; i++) {
				lightTexture.pixels[i] = lightColor; 
			}
			lightTexture.mask(lightMask);
			lightTexture.updatePixels();
			
			//fill the lightPoolTexture with color
			lightPoolTexture.loadPixels();
			for (int i = 0; i < lightPoolTexture.pixels.length; i++) {
				lightPoolTexture.pixels[i] = lightColor; 
			}
			lightPoolTexture.mask(lightPoolMask);
			lightPoolTexture.updatePixels();
			
			lightModel.setTexture(lightTexture);
			lightModel.enableTexture();
			lightPoolModel.setTexture(lightPoolTexture);
			lightPoolModel.enableTexture();
		}
	  
	}

	//setLightColor rebuilds the PImage texture for a given light
	//Intensity helps adjust things visually, not part of the real MetalMatisse functions 
    @Override
	public void setLightColor(String nameOfLight, Color lightColor) {

        LightObject light = lights.get(nameOfLight);

        if (light != null){

            int r = Math.min(255, (int) (lightColor.getRed() * light.lightIntensity));
            int g = Math.min(255, (int) (lightColor.getGreen() * light.lightIntensity));
            int b = Math.min(255, (int) (lightColor.getBlue() * light.lightIntensity));

            light.lightPoolTexture.loadPixels();
            for (int i = 0; i < light.lightPoolTexture.pixels.length; i++) {
                light.lightPoolTexture.pixels[i] = color(lightColor.getRed(),
                                                         lightColor.getGreen(),
                                                         lightColor.getBlue()); 
            }

            light.lightPoolTexture.mask(light.lightPoolMask);
            light.lightPoolTexture.updatePixels();

            light.lightTexture.loadPixels();
            for (int i = 0; i < light.lightTexture.pixels.length; i++) {
                light.lightTexture.pixels[i] = color(r,g,b); 
            }

            light.lightTexture.mask(light.lightMask);
            light.lightTexture.updatePixels();
        }
	}

	//setSensorState does what it says on the tin
	@Override
	public void setSensorState(String sensorName, boolean isOn) {
		LightObject sensor = lights.get(sensorName);
		sensor.triggerState = isOn;
		System.err.println("Trigger " + sensorName + "is tripped high: " + sensor.triggerState);

	}
	
	//declare and set some display flags
	boolean showParticipants = true;
	boolean showSensorBeams = true;
	boolean showEnvironment = true;
	boolean bTexture = true;
	boolean bStroke = false;
	int zoomLevel = 1200;

	//declare all OBJModels
	OBJModel sculptureVase, sculptureSolid, sculptureScreen, environment, plane, blankOBJ, volumeB01, volumeB02, volumeB03, volumeC01A, volumeC01B, volumeC02A, volumeC02B, volumeC03A, volumeC03B, volumeF01, volumeF02, volumeF03, volumeF05, volumeF06, volumeF08, volumeF09, volumeF10, volumeF11, volumeF12, volumeL01, volumeL02, volumeV01, volumeV02, volumeV03, volumeV04, modelpB01, modelpB02, modelpB03, modelpF01, modelpF02, modelpF03, modelpF05, modelpF06, modelpF08, modelpF09, modelpF10, modelpF11, modelpF12, modelsB01, modelsB02, modelsB03, modelsF01, modelsF02, modelsF03, modelsF05, modelsF06, modelsF08, modelsF09, modelsF10, modelsF11, modelsF12, modelsT01, modeldC01, modeldC02, modeldC03, modeldF01, modeldF02, modeldF03, modeldF05, modeldF06, modeldF08, modeldF09, modeldF10, modeldF11, modeldF12;

	//declare some textures
	PImage beamTexture, vaseTexture, vaseMask;
	
	
	//Declare the lights hashmap
	HashMap<String,LightObject> lights = new HashMap<String,LightObject>();

	VizOSCReceiver client;

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
	  sculptureVase = new OBJModel(this, "../depends/models/sculptureVase.obj", "absolute", TRIANGLES);
	  sculptureSolid = new OBJModel(this, "../depends/models/sculptureSolid.obj", "absolute", TRIANGLES);
	  sculptureScreen = new OBJModel(this, "../depends/models/sculptureScreen.obj", "absolute", TRIANGLES);
	  environment = new OBJModel(this, "../depends/models/environment.obj", "absolute", TRIANGLES);
	  plane = new OBJModel(this, "../depends/models/plane.obj", "absolute", TRIANGLES);
	  blankOBJ = new OBJModel(this, "../depends/models/blankOBJ.obj", "absolute", TRIANGLES);
	  volumeB01 = new OBJModel(this, "../depends/models/B01.obj", "absolute", POLYGON);
	  volumeB02 = new OBJModel(this, "../depends/models/B02.obj", "absolute", POLYGON);
	  volumeB03 = new OBJModel(this, "../depends/models/B03.obj", "absolute", POLYGON);
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
	  modelsT01 = new OBJModel(this, "../depends/models/sT01.obj", "absolute", TRIANGLES);
	  modeldC01 = new OBJModel(this, "../depends/models/dC01.obj", "absolute", TRIANGLES);
	  modeldC02 = new OBJModel(this, "../depends/models/dC02.obj", "absolute", TRIANGLES);
	  modeldC03 = new OBJModel(this, "../depends/models/dC03.obj", "absolute", TRIANGLES);
	  modeldF01 = new OBJModel(this, "../depends/models/dF01.obj", "absolute", TRIANGLES);
	  modeldF02 = new OBJModel(this, "../depends/models/dF02.obj", "absolute", TRIANGLES);
	  modeldF03 = new OBJModel(this, "../depends/models/dF03.obj", "absolute", TRIANGLES);
	  modeldF05 = new OBJModel(this, "../depends/models/dF05.obj", "absolute", TRIANGLES);
	  modeldF06 = new OBJModel(this, "../depends/models/dF06.obj", "absolute", TRIANGLES);
	  modeldF08 = new OBJModel(this, "../depends/models/dF08.obj", "absolute", TRIANGLES);
	  modeldF09 = new OBJModel(this, "../depends/models/dF09.obj", "absolute", TRIANGLES);
	  modeldF10 = new OBJModel(this, "../depends/models/dF10.obj", "absolute", TRIANGLES);
	  modeldF11 = new OBJModel(this, "../depends/models/dF11.obj", "absolute", TRIANGLES);
	  modeldF12 = new OBJModel(this, "../depends/models/dF12.obj", "absolute", TRIANGLES);
	  
	  //Build Vase's hole texture
	  vaseTexture = createImage(512, 512, RGB);
	  vaseTexture.loadPixels();
	  for (int i = 0; i < vaseTexture.pixels.length; i++) {
		  vaseTexture.pixels[i] = color(230, 230, 240); 
	  }
	  vaseMask = loadImage("../depends/models/holes.jpg");
	  vaseTexture.mask(vaseMask);
	  vaseTexture.updatePixels();
	  sculptureVase.setTexture(vaseTexture);
	  sculptureVase.enableTexture();
	  
	  //Build sensorBeam's texture
	  beamTexture = createImage(128,128, RGB);
	  beamTexture.loadPixels();
	  for (int i = 0; i < beamTexture.pixels.length; i++) {
		  beamTexture.pixels[i] = color(255, 0, 0); 
	  }
	  beamTexture.updatePixels();

	  
	  //initialize LightObjects
	  LightObject B01 = new LightObject(volumeB01, modelpB01, modelsB01, blankOBJ, 0.4);
	  LightObject B02 = new LightObject(volumeB02, modelpB02, modelsB02, blankOBJ, 0.4);
	  LightObject B03 = new LightObject(volumeB03, modelpB03, modelsB03, blankOBJ, 0.4);
	  LightObject C01A = new LightObject(volumeC01A, blankOBJ, blankOBJ, modeldC01, 0.25);
	  LightObject C01B = new LightObject(volumeC01B, blankOBJ, blankOBJ, blankOBJ, 0.25);
	  LightObject C02A = new LightObject(volumeC02A, blankOBJ, blankOBJ, modeldC02, 0.25);
	  LightObject C02B = new LightObject(volumeC02B, blankOBJ, blankOBJ, blankOBJ, 0.25);
	  LightObject C03A = new LightObject(volumeC03A, blankOBJ, blankOBJ, modeldC03, 0.25);
	  LightObject C03B = new LightObject(volumeC03B, blankOBJ, blankOBJ, blankOBJ, 0.25);
	  LightObject F01 = new LightObject(volumeF01, modelpF01, modelsF01, modeldF01, 1.0);
	  LightObject F02 = new LightObject(volumeF02, modelpF02, modelsF02, modeldF02, 1.0);
	  LightObject F03 = new LightObject(volumeF03, modelpF03, modelsF03, modeldF03, 1.0);
	  LightObject F05 = new LightObject(volumeF05, modelpF05, modelsF05, modeldF05, 1.0);
	  LightObject F06 = new LightObject(volumeF06, modelpF06, modelsF06, modeldF06, 1.0);
	  LightObject F08 = new LightObject(volumeF08, modelpF08, modelsF08, modeldF08, 1.0);
	  LightObject F09 = new LightObject(volumeF09, modelpF09, modelsF09, modeldF09, 1.0);
	  LightObject F10 = new LightObject(volumeF10, modelpF10, modelsF10, modeldF10, 1.0);
	  LightObject F11 = new LightObject(volumeF11, modelpF11, modelsF11, modeldF11, 1.0);
	  LightObject F12 = new LightObject(volumeF12, modelpF12, modelsF12, modeldF12, 1.0);
	  LightObject L01 = new LightObject(volumeL01, blankOBJ, blankOBJ, blankOBJ, 1.0);
	  LightObject L02 = new LightObject(volumeL02, blankOBJ, blankOBJ, blankOBJ, 1.0);
	  LightObject V01 = new LightObject(volumeV01, blankOBJ, blankOBJ, blankOBJ, 1.0);
	  LightObject V02 = new LightObject(volumeV02, blankOBJ, blankOBJ, blankOBJ, 1.0);
	  LightObject V03 = new LightObject(volumeV03, blankOBJ, blankOBJ, blankOBJ, 1.0);
	  LightObject V04 = new LightObject(volumeV04, blankOBJ, blankOBJ, blankOBJ, 1.0);
	  LightObject T01 = new LightObject(blankOBJ, blankOBJ, modelsT01, blankOBJ, 1.0);
	  
	  //fill the lights hashmap
	  lights.put("b01", B01);
	  lights.put("b02", B02);
	  lights.put("b03", B03);
	  lights.put("c01A", C01A);
	  lights.put("c01B", C01B);
	  lights.put("c02A", C02A);
	  lights.put("c02B", C02B);
	  lights.put("c03A", C03A);
	  lights.put("c03B", C03B);
	  lights.put("f01", F01);
	  lights.put("f02", F02);
	  lights.put("f03", F03);
	  lights.put("f05", F05);
	  lights.put("f06", F06);
	  lights.put("f08", F08);
	  lights.put("f09", F09);
	  lights.put("f10", F10);
	  lights.put("f11", F11);
	  lights.put("f12", F12);
	  lights.put("l01", L01);
	  lights.put("l02", L02);
	  lights.put("v01", V01);
	  lights.put("v02", V02);
	  lights.put("v03", V03);
	  lights.put("v04", V04);
	  lights.put("t01", T01);
	  

	  //Set stroke color to white, then hide strokes
	  stroke(255);
	  noStroke();

	  // listener for Norfolk
	  client = new VizOSCReceiver();
	  client.load(new ElectrolandProperties("osc.properties"));
	  client.addListener(this);
	  client.start();
	  Runtime.getRuntime().addShutdownHook(new ShutdownThread(this));
	}


	public void draw()
	{
		background(17, 24, 34);
		//lights();
		//Add some lighting
		pointLight(128, 128, 128, 0, 50, 2000);
		pointLight(128, 128, 128, 0, 50, -2000);
		pointLight(128, 128, 128, 2000, 50, 0);
		pointLight(128, 128, 128, -2000, 50, 0);

		shininess(1);
		sculptureSolid.draw();
		sculptureScreen.draw();
		shininess(0);

		if (showEnvironment == true) {
			environment.draw();
		}
		else {
			plane.draw();
		}

		//iterate through allLightObjects and draw solid stuff
		for (LightObject light : lights.values()) {
			if (showSensorBeams == true) {
				if (light.triggerState == true) light.sensorBeamModel.draw();
			}
			if (showParticipants == true) {
				if (light.triggerState == true) light.participantModel.draw();
			}
		}

		//iterate through allLightObjects and draw light cones in ADD mode
		blendMode(ADD);
		for (LightObject light : lights.values()) {
			light.lightPoolModel.draw();
			light.lightModel.draw();
		}
		blendMode(BLEND);
		sculptureVase.draw();

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
			for (String light : lights.keySet()) {
				int rRand = rand.nextInt(255);
				int gRand = rand.nextInt(255);
				int bRand = rand.nextInt(255);
				Color color = new Color(rRand, gRand, bRand);
				setLightColor(light, color);
				
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
			if(!lights.get("B01").triggerState) {
				setSensorState("B01", true);
	        } 
	        else {
	        	setSensorState("B01", false);
	        }

		}
		if(key == '2') {
			if(!lights.get("B02").triggerState) {
				setSensorState("B02", true);
	        } 
	        else {
	        	setSensorState("B02", false);
	        }

		}
			
		if(key == '3') {
			if(!lights.get("F01").triggerState) {
				setSensorState("F01", true);
	        } 
	        else {
	        	setSensorState("F01", false);
	        }

		}
			
		if(key == '4') {
			if(!lights.get("F02").triggerState) {
				setSensorState("F02", true);
	        } 
	        else {
	        	setSensorState("F02", false);
	        }

		}
		
		if(key == '5') {
			if(!lights.get("F03").triggerState) {
				setSensorState("F03", true);
	        } 
	        else {
	        	setSensorState("F03", false);
	        }

		}
		
		if(key == '6') {
			if(!lights.get("F05").triggerState) {
				setSensorState("F05", true);
	        } 
	        else {
	        	setSensorState("F05", false);
	        }

		}
			

			
	}

    @Override
    public void shutdown() {
        client.stop();
    }
}