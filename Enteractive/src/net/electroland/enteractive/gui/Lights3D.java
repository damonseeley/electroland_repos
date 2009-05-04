package net.electroland.enteractive.gui;

import java.util.ListIterator;

import net.electroland.enteractive.core.Model;
import net.electroland.lighting.detector.Detector;
import net.electroland.lighting.detector.Recipient;
import processing.core.PApplet;

@SuppressWarnings("serial")
public class Lights3D extends PApplet{

	private int width, height;				// applet dimensions
	private int floorWidth, floorHeight;	// tile grid
	private int faceWidth, faceHeight;		// light grid
	private float rotX, rotY, velX, velY;	// rotation properties
	private float zoom = 2.0f;
	private boolean dampening = true;
	private int tileSize = 10;
	private Recipient floor, face;
	private boolean faceMode = false;
	private Model m;
	
	public Lights3D(int width, int height, Recipient floor, Recipient face, Model m){
		this.width = width;
		this.height = height;
		this.floor = floor;
		this.face = face;
		this.m = m;
		floorWidth = 16;
		floorHeight = 11;
		faceWidth = 18;
		faceHeight = 6;
		rotX = -70;
		rotY = -30;
		velX = 0;
		velY = 0;
	}
	
	public void setup(){
		size(width, height, P3D);
		frameRate(30);
	}
	
	public void draw(){
		background(0);
		translate(width/2, height/1.5f);
		rotateY(-radians(rotY));
		rotateX(-radians(rotX));
		scale(zoom);
		noFill();
		translate(-tileSize*floorWidth/2, -tileSize*floorHeight/2);
		drawFloor();
		drawSensors();
		translate(-12,0,150);
		drawFace();
		  
		rotX += velX;
		rotY += velY;
		if(dampening){
			velX *= 0.95f;
		    velY *= 0.95f;
		}
		if(mousePressed){
		    if(mouseButton == LEFT){
		    	velX += (mouseY-pmouseY) * 0.01f;
		    	velY -= (mouseX-pmouseX) * 0.01f;
		    } else if(mouseButton == RIGHT){
		    	zoom += (mouseY-pmouseY) *0.001f;
		    }
		  }
	}
	
	public void setMode(int mode){
		if(mode == 1){
			faceMode = false;
		} else if (mode == 2){
			faceMode = true;
		}
	}
	
	public void drawFace(){
		rotateX(radians(-90));
		if(faceMode){
			rotateY(radians(-90));
			translate(-tileSize*floorWidth/4, -tileSize*floorWidth/2, 0);
		}

		fill(255,255,255,20);
		translate(0,0,-1);
		rect(-1, -1, faceWidth*12, faceHeight*24 - 12);
		translate(0,0,1);
		
		try{
			ListIterator<Detector> i = face.getDetectorPatchList().listIterator();
			int channel = 0;
			noFill();
			while(i.hasNext()){
				Detector d = i.next();
				//System.out.print("value for channel " + (channel++) + "=");
				if (d != null){
					int val = face.getLastEvaluatedValue(d) & 0xFF;
					int x = channel % faceWidth;
					int y = channel / faceWidth;
					stroke(val,0,0);
					rect(x*12, y*24, 10, 10);
					//System.out.println(channel +" "+ val);
				}else{
					//System.out.println("- no detector -");
				}
				channel++;
			}
		} catch(NullPointerException e){
			e.printStackTrace();
		}
		
		/*
		for(int y=0; y<faceHeight; y++){
			for(int x = 0; x<faceWidth; x++){
				rect(x*12, y*24, 10, 10);
			}
		}
		*/
		
	}
	
	public void drawFloor(){
		
		try{
			ListIterator<Detector> i = floor.getDetectorPatchList().listIterator();
			int channel = 0;
			fill(255,255,255,20);
			translate(0,0,-1);
			rect(-1, -1, floorWidth*12, floorHeight*12);
			translate(0,0,1);
			noFill();
			while(i.hasNext()){
				Detector d = i.next();
				//System.out.print("value for channel " + (channel++) + "=");
				if (d != null){
					int val = floor.getLastEvaluatedValue(d) & 0xFF;
					int x = channel % floorWidth;
					int y = channel / floorWidth;
					stroke(val,0,0);
					rect(x*12, y*12, 10, 10);
					//System.out.println(val);
				}else{
					//System.out.println("- no detector -");
				}
				channel++;
			}
		} catch(NullPointerException e){
			e.printStackTrace();
		}
	
		/*
		for(int y=0; y<floorHeight; y++){
			for(int x = 0; x<floorWidth; x++){
				rect(x*12, y*12, 10, 10);
			}
		}
		*/
		
	}
	
	public void drawSensors(){
		boolean[] sensors;
		synchronized(m){
			sensors = m.getSensors();
		}
		
		if(sensors != null){
			noStroke();
			for(int i=0; i<sensors.length; i++){
				if(sensors[i]){
					fill(255,255,255);
					int x = i % floorWidth;
					int y = i / floorWidth;
					rect(x*12 + 4, y*12 + 4, 2, 2);
				}
			}
		}
	}
	
}
