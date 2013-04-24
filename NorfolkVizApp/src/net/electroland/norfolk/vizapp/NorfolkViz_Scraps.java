package net.electroland.norfolk.vizapp;
import processing.core.*;
import saito.objloader.*;
import shapes3d.utils.*;
import shapes3d.animation.*;
import shapes3d.*;

public class NorfolkViz_Scraps extends PApplet {


	//declare all PShapes globally
	OBJModel sculpture, environment;
	Cone lightCone;

	public void setup()
	{
	  
	  //Scene instantiation
	  size(800, 600, P3D);

	  //Define all objects and their geo  
	  sculpture = new OBJModel(this, "../depends/models/sculpture.obj", "absolute", TRIANGLES);
	  environment = new OBJModel(this, "../depends/models/environment.obj", "absolute", TRIANGLES);
	  lightCone = new Cone(this, 24);
	  lightCone.setSize(140, 140, 500);
	  lightCone.moveTo(new PVector(500, 500, 0));
	  lightCone.fill(color(255, 255, 0), S3D.ALL);
	  lightCone.drawMode(S3D.SOLID, S3D.ALL);
	  
	  stroke(255);
	  noStroke();
	}



	public void draw()
	{
	    background(129);
	    lights();

	    sculpture.draw();
	    environment.draw();
	    lightCone.draw();
	    
	      
	    }
	    
}
