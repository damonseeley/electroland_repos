package net.electroland.enteractive.gui;

import processing.core.PApplet;
import processing.core.PConstants;

@SuppressWarnings("serial")
public class Lights3D extends PApplet{

	private int width, height;				// applet dimensions
	private int floorWidth, floorHeight;	// tile grid
	private int faceWidth, faceHeight;		// light grid
	private float rotX, rotY, velX, velY;	// rotation properties
	private float zoom = 1.8f;
	private boolean dampening = true;
	private int tileSize = 10;
	
	public Lights3D(int width, int height){
		this.width = width;
		this.height = height;		
		floorWidth = 16;
		floorHeight = 11;
		faceWidth = 18;
		faceHeight = 6;
		rotX = -70;
		rotY = 30;
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
		pushMatrix();
		stroke(255,0,0);
		noFill();
		drawFloor();
		popMatrix();
		pushMatrix();
		stroke(255,0,0);
		noFill();
		drawFace();
		popMatrix();
		  
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
	
	public void drawFace(){
		rotateX(radians(90));
		translate(-tileSize*faceWidth/2, tileSize*faceHeight/2, tileSize*floorHeight/2);
		for(int y=0; y<faceHeight; y++){
			for(int x = 0; x<faceWidth; x++){
				rect(x*12, y*24, 10, 10);
			}
		}
	}
	
	public void drawFloor(){
		translate(-tileSize*floorWidth/2, -tileSize*floorHeight/2);
		for(int y=0; y<floorHeight; y++){
			for(int x = 0; x<floorWidth; x++){
				rect(x*12, y*12, 10, 10);
			}
		}
	}
	
}
