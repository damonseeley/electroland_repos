package net.electroland.norfolk.vizapp;
import processing.core.*;
import saito.objloader.*;
import shapes3d.utils.*;
import shapes3d.animation.*;
import shapes3d.*;

public class NorfolkViz_Scraps extends PApplet {


	//declare all PShapes globally
	OBJModel sculpture, environment;
	PImage gradTexture;
	PImage gradMask;
	Cone lightCone;

	public void setup()
	{
	  
	  //Scene instantiation
	  size(800, 600, P3D);
	  
	  //Define that texture
	  gradMask = loadImage("../depends/models/lightGrad.jpg");
	  gradTexture = createImage(512,512,RGB);
	  for (int i = 0; i < gradTexture.pixels.length; i++) {
		  gradTexture.pixels[i] = color(255, 90, 102); 
		}
	  gradTexture.mask(gradMask);

	  //Define all objects and their geo  
	  sculpture = new OBJModel(this, "../depends/models/sculpture.obj", "absolute", TRIANGLES);
	  environment = new OBJModel(this, "../depends/models/environment.obj", "absolute", TRIANGLES);
	  lightCone = new Cone(this, 24);
	  lightCone.setSize(140, 140, 500);
	  lightCone.moveTo(new PVector(500, 500, 0));
	  lightCone.setTexture(gradTexture);
	  //lightCone.drawMode(S3D.SOLID, S3D.ALL);
	  
	  stroke(255);
	  noStroke();
	}



	public void draw()
	{
	    background(129);
	    lights();

	    sculpture.draw();
	    environment.draw();
	    //blendMode(SCREEN);
	    lightCone.draw();
	    //image(gradTexture, 0, 0);
	      
	    }
	    
}
