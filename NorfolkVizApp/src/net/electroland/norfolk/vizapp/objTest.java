package net.electroland.norfolk.vizapp;
import processing.core.*;
import saito.objloader.*;


public class objTest extends PApplet {
	
	/**
	 * Load and Display an OBJ Shape. 
	 * 
	 * The loadShape() command is used to read simple SVG (Scalable Vector Graphics)
	 * files and OBJ (Object) files into a Processing sketch. This example loads an
	 * OBJ file of a rocket and displays it to the screen. 
	 */


	OBJModel rocket;

	float ry;
	  
	public void setup() {
	  size(640, 360, P3D);
	    
	  rocket = new OBJModel(this, "../depends/models/environment.obj", "absolute", TRIANGLES);
	  rocket.enableTexture();
	  //Set stroke color to white, then hide strokes  
	  stroke(255);
	  noStroke();
	}

	public void draw() {
	  background(0);
	  lights();
	  
	  translate(width/2, height/2 + 100, -200);
	  rotateZ(PI);
	  rotateY(ry);
	  rocket.draw();
	  
	  ry += 0.02;
	}
}