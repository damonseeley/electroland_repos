package net.electroland.connection.ui;

import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Person;
import processing.core.PApplet;
import processing.core.PFont;
import controlP5.ControlEvent;
import controlP5.ControlP5;
import controlP5.Radio;
import controlP5.Textlabel;
import java.util.Iterator;

public class GUI extends PApplet{
	
	private static final long serialVersionUID = 1L;
	public int w, h;
	public int gridx, gridy;
	public ControlP5 controls;
	public PFont arial;
	public boolean drawingOn;
	public int xcm, ycm;			// grid size in cm
	public int xoffcm, yoffcm;		// grid offset in cm
	public int nudgesize;			// amount to nudge
	public Textlabel population;
	public Textlabel trackingduration;
	
	public GUI(int width, int height){
		w = width;
		h = height;
		gridx = 28;	// horizontal mode
		gridy = 6;
		xcm = Integer.parseInt(ConnectionMain.properties.get("gridWidth"));
		ycm = Integer.parseInt(ConnectionMain.properties.get("gridHeight"));
		xoffcm = 0;
		yoffcm = 0;
		nudgesize = 1;
	}
	
	public void setup(){
		size(w,h);
		frameRate(15);
		drawingOn = true;
		arial = createFont("Arial", 10, false);
		smooth();
		noStroke();
		controls = new ControlP5(this);
		Radio r = controls.addRadio("displaymode",590,15);
		r.setColorActive(255);
		r.addItem("conductor mode",-1);
		r.addItem("vegas",0);
		r.addItem("wave",1);
		r.addItem("musicbox",2);
		r.addItem("matrix",3);
		r.addItem("biggest",4);
		r.addItem("screensaver",-2);

		Radio l = controls.addRadio("littleshows",700,15);
		l.setColorActive(255);
		l.addItem("randomfill", 0);
		l.addItem("quickblue", 1);
		l.addItem("quickred", 2);
		l.addItem("quickpurple", 3);
		
		controls.addSlider("red_fade_length",0.01f,0.95f,Float.parseFloat(ConnectionMain.properties.get("redFadeLength")),790,15,100,10);	// red light fade length
		controls.addSlider("blue_fade_length",0.01f,0.95f,Float.parseFloat(ConnectionMain.properties.get("blueFadeLength")),790,29,100,10);	// blue light fade length
		controls.addSlider("forward_compensation",0f,20f,Float.parseFloat(ConnectionMain.properties.get("forwardCompensation")),790,43,100,10);	// forward movement compensation
		controls.addTextlabel("poplabel","CURRENT POPULATION:",790,60);
		population = new Textlabel(this, "0", 900, 60, 50, 20, 0xffffff, ControlP5.standard56);
		controls.addTextlabel("trackinglabel","TRACKING DURATION:",790,80);
		trackingduration = new Textlabel(this, "?", 900, 80, 50, 20, 0xffffff, ControlP5.standard56);
		// sliders
		controls.addSlider("grid_width",4000,7000,Integer.parseInt(ConnectionMain.properties.get("gridWidth")),590,150,300,10);				// grid property sliders
		controls.addSlider("grid_height",600,1600,Integer.parseInt(ConnectionMain.properties.get("gridHeight")),590,164,300,10);
		controls.addSlider("x_offset",-2000,2000,Integer.parseInt(ConnectionMain.properties.get("xoffset")),590,178,300,10);
		controls.addSlider("y_offset",-2000,2000,Integer.parseInt(ConnectionMain.properties.get("yoffset")),590,192,300,10);
		controls.addSlider("min_connection",0,10,4,590,206,300,10);		// minimum distance for connection before joining
		controls.addSlider("max_connection",0,20,10,590,220,300,10);	// max distance for connection before breaking
		controls.addSlider("skew",-1000,1000,0,590,234,300,10);			// skew
		// toggles
		controls.addToggle("x_flip", Boolean.parseBoolean(ConnectionMain.properties.get("flipX")), 590, 250, 10, 10).setColorActive(255);	// grid property toggles
		controls.addToggle("y_flip", Boolean.parseBoolean(ConnectionMain.properties.get("flipY")), 640, 250, 10, 10).setColorActive(255);
		controls.addToggle("swap_x_and_y", Boolean.parseBoolean(ConnectionMain.properties.get("swapXY")), 690, 250, 10, 10).setColorActive(255);
		controls.addToggle("dashed_line", Boolean.parseBoolean(ConnectionMain.properties.get("dashedLine")), 790, 250, 10, 10).setColorActive(255);
		controls.addToggle("connection_mode", ConnectionMain.renderThread.conductor.connectionMode, 880, 250, 10, 10).setColorActive(255);
		//controls.addToggle("audio", Boolean.parseBoolean(ConnectionMain.properties.get("audio")), 590, 285, 10, 10).setColorActive(255);
		//controls.addToggle("print_offset_and_scaling_values", false, 640, 285, 10, 10).setColorActive(255);
	}
	
	public void controlEvent(ControlEvent e){	// this catches all the control widget activity
		if(e.controller().name() == "grid_width"){
			xcm = (int)e.controller().value();
			ConnectionMain.personTracker.xscale = 1/(float)e.controller().value();
		} else if(e.controller().name() == "grid_height"){
			ycm = (int)e.controller().value();
			ConnectionMain.personTracker.yscale = 1/(float)e.controller().value();
		} else if(e.controller().name() == "x_offset"){
			xoffcm = (int)e.controller().value();
			ConnectionMain.personTracker.yoffset = (float)e.controller().value()/xcm;
		} else if(e.controller().name() == "y_offset"){
			yoffcm = (int)e.controller().value();
			ConnectionMain.personTracker.xoffset = (float)e.controller().value()/ycm;
		} else if(e.controller().name() == "skew"){
			ConnectionMain.properties.put("skew", str(e.controller().value()));
		} else if(e.controller().name() == "red_fade_length"){
			ConnectionMain.properties.put("redFadeLength", str(e.controller().value()));
		} else if(e.controller().name() == "blue_fade_length"){
			ConnectionMain.properties.put("blueFadeLength", str(e.controller().value()));
		} else if(e.controller().name() == "forward_compensation"){
			ConnectionMain.properties.put("forwardCompensation", str(e.controller().value()));
		} else if(e.controller().name() == "displaymode"){
			ConnectionMain.renderThread.conductor.setMode((int)e.controller().value());			
		} else if(e.controller().name() == "littleshows"){
			if(frameCount > 5){
				ConnectionMain.renderThread.conductor.playLittleShow((int)e.controller().value());
			}
		}
	}
	
	public void x_flip(boolean value){
		ConnectionMain.personTracker.xflip = value;
	}
	
	public void y_flip(boolean value){
		ConnectionMain.personTracker.yflip = value;
	}
	
	public void swap_x_and_y(boolean value){
		ConnectionMain.personTracker.swap = value;
	}
	
	public void dashed_line(boolean value){
		ConnectionMain.renderThread.conductor.trackingConnections.dashedMode = value;
	}
	
	public void connection_mode(boolean value){
		ConnectionMain.renderThread.conductor.connectionMode = value;
		ConnectionMain.renderThread.conductor.setMode(-1);
	}
	
	public void audio(boolean value){
		ConnectionMain.soundController.audioEnabled = value;
	}
	
	public void print_offset_and_scaling_values(boolean value){
		System.out.println("long side: " + xcm + "cm, short side: " + ycm + "cm, xoffset: " + xoffcm + "cm, yoffset: " + yoffcm +"cm");
	}
	
	public void draw(){
		background(0);
		if(drawingOn){
			pushMatrix();
			translate(15,15);
			drawSquareGrid();
			drawLightGrid();
			popMatrix();
		}
		pushMatrix();
		translate(15, 150);
		drawSquareGrid();
		try{
			drawPeople();
		} catch(NullPointerException e){
			System.err.println("gui error");
			e.printStackTrace();
		}
		popMatrix();
		population.setValue(String.valueOf(ConnectionMain.personTracker.peopleCount()));
		population.draw(this);
		trackingduration.setValue(String.valueOf(ConnectionMain.renderThread.conductor.getTrackingDuration()/1000)+" seconds");
		trackingduration.draw(this);
		//pushMatrix();
		//translate(15, 285);
		//drawControllers();
		//popMatrix();
	}
	
	public void drawControllers(){
		noStroke();
		textFont(arial);
		for(int i=0; i<14; i++){
			fill(0, 255, 0);	// color based on controller status
			rect(i*40 + 12, 55, 16, 16);
			int ip = 21 + i;
			fill(150);
			textAlign(LEFT);
			text(str(i*24), i*40 + 2, 40);			// offsetA
			fill(0);
			textAlign(CENTER);
			text(str(ip), i*40 + 20, 65);			// IP address
			fill(150);
			textAlign(RIGHT);
			text(str(i*24 + 12), i*40 + 38, 100);	// offsetB
			stroke(50);
			line(i*40 + 40, 0, i*40 + 40, gridy*20);
			noStroke();
		}
		stroke(100);
		noFill();
		rect(0, 0, gridx*20, gridy*20);
	}
	
	public void drawLightGrid(){
		int light = 0;
		stroke(100);
		noFill();
		rect(0, 0, gridx*20, gridy*20);
		noStroke();
		// horizontal mode
		byte[] buffer = ConnectionMain.renderThread.getBuffer();
		for(int x=0; x<gridx; x++){
			for(int y=gridy-1; y>=0; y--){
				//fill((int)ConnectionMain.renderThread.buffer[light*2 + 2] & 0xFF, 0, (int)ConnectionMain.renderThread.buffer[light*2 + 3] & 0xFF);
				fill((int)buffer[light*2 + 2] & 0xFF, 0, (int)buffer[light*2 + 3] & 0xFF);
				ellipse((x*20)+10, (y*20)+10, 15, 15);
				light++;
			}
		}
	}

	public void drawPeople(){	// this has the tendency to crash from nullpointerexceptions
		noStroke();
		//Iterator<Person> i = ConnectionMain.renderThread.lightController.peopleCollection.iterator();
		Iterator<Person> i = ConnectionMain.personTracker.getPersonIterator();
		while (i.hasNext()){
			fill(255);
			Person person = i.next();
			float loc[] = person.getFloatLoc();
			ellipse(loc[1]*20, gridy*20 - loc[0]*20, 8, 8);					
		}
	}
	
	public void drawSquareGrid(){
		stroke(50);
		noFill();
		for(int i=1; i<gridx; i++){
			line(i*20, 0, i*20, gridy*20);
		}
		for(int i=1; i<gridy; i++){
			line(0, i*20, gridx*20, i*20);
		}
		stroke(100);
		rect(0, 0, gridx*20, gridy*20);
	}
	
	public void keyPressed(){
		//println(key +"="+ keyCode);
		if(keyCode == 87){										// W
			xoffcm += nudgesize;
			ConnectionMain.personTracker.xoffset = xoffcm/(float)ycm;		
		} else if(keyCode == 83){								// S
			xoffcm -= nudgesize;
			ConnectionMain.personTracker.xoffset = xoffcm/(float)ycm;
		} else if(keyCode == 65){								// A
			yoffcm -= nudgesize;
			ConnectionMain.personTracker.yoffset = yoffcm/(float)xcm;
		} else if(keyCode == 68){								// D
			yoffcm += nudgesize;
			ConnectionMain.personTracker.yoffset = yoffcm/(float)xcm;
		} else if(keyCode == 37){								// left
			xcm -= nudgesize;
			ConnectionMain.personTracker.xscale = 1/(float)xcm;
		} else if(keyCode == 38){								// up
			ycm += nudgesize;
			ConnectionMain.personTracker.yscale = 1/(float)ycm;
		} else if(keyCode == 39){								// right
			xcm += nudgesize;
			ConnectionMain.personTracker.xscale = 1/(float)xcm;
		} else if(keyCode == 40){								// down
			ycm -= nudgesize;
			ConnectionMain.personTracker.yscale = 1/(float)ycm;
		} else if(keyCode == 61){								// plus
			nudgesize += 1;
			println("nudge size " + nudgesize + "cm");
		} else if(keyCode == 45){								// minus
			if(nudgesize > 1){
				nudgesize -= 1;
				println("nudge size " + nudgesize + "cm");
			}
		}
	}
}
