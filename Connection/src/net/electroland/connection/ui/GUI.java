package net.electroland.connection.ui;

import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Person;
import net.electroland.connection.core.SoundController;
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
		r.addItem("blank animation",-3);

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
		
		// bangs for audio testing
		controls.addBang("speaker1", 30, 100, 10, 10).setLabel("S1");
		controls.addBang("speaker2", 30, 40, 10, 10).setLabel("S2");
		controls.addBang("speaker3", 70, 100, 10, 10).setLabel("S3");
		controls.addBang("speaker4", 70, 40, 10, 10).setLabel("S4");
		controls.addBang("speaker5", 110, 100, 10, 10).setLabel("S5");
		controls.addBang("speaker6", 110, 40, 10, 10).setLabel("S6");
		controls.addBang("speaker7", 150, 100, 10, 10).setLabel("S7");
		controls.addBang("speaker8", 150, 40, 10, 10).setLabel("S8");
		controls.addBang("speaker9", 190, 100, 10, 10).setLabel("S9");
		controls.addBang("speaker10", 190, 40, 10, 10).setLabel("S10");
		controls.addBang("speaker11", 230, 100, 10, 10).setLabel("S11");
		controls.addBang("speaker12", 230, 40, 10, 10).setLabel("S12");
		controls.addBang("speaker13", 270, 100, 10, 10).setLabel("S13");
		controls.addBang("speaker14", 270, 40, 10, 10).setLabel("S14");
		controls.addBang("speaker15", 310, 100, 10, 10).setLabel("S15");
		controls.addBang("speaker16", 310, 40, 10, 10).setLabel("S16");
		controls.addBang("speaker17", 350, 100, 10, 10).setLabel("S17");
		controls.addBang("speaker18", 350, 40, 10, 10).setLabel("S18");
		controls.addBang("speaker19", 390, 100, 10, 10).setLabel("S19");
		controls.addBang("speaker20", 390, 40, 10, 10).setLabel("S20");
		controls.addBang("speaker21", 430, 100, 10, 10).setLabel("S21");
		controls.addBang("speaker22", 430, 40, 10, 10).setLabel("S22");
		controls.addBang("speaker23", 470, 100, 10, 10).setLabel("S23");
		controls.addBang("speaker24", 470, 40, 10, 10).setLabel("S24");
		controls.addBang("speaker25", 510, 100, 10, 10).setLabel("S25");
		controls.addBang("speaker26", 510, 40, 10, 10).setLabel("S26");
		controls.addBang("speaker27", 550, 100, 10, 10).setLabel("S27");
		controls.addBang("speaker28", 550, 40, 10, 10).setLabel("S28");
		
		// set all audio buttons to invisible by default
		for(int i=1; i<29; i++){
			controls.controller("speaker"+i).setVisible(false);
		}
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
			if((int)e.controller().value() == -3){
				for(int i=1; i<29; i++){
					controls.controller("speaker"+i).setVisible(true);
				}
			} else {
				for(int i=1; i<29; i++){
					controls.controller("speaker"+i).setVisible(false);
				}
			}
		} else if(e.controller().name() == "littleshows"){
			if(frameCount > 5){
				ConnectionMain.renderThread.conductor.playLittleShow((int)e.controller().value());
			}
		} else if(e.controller().name() == "speaker1"){
			System.out.println("speaker1 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 0, 0.8f, "speaker 1 test");
		} else if(e.controller().name() == "speaker2"){
			System.out.println("speaker2 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 0, 0.8f, "speaker 2 test");
		} else if(e.controller().name() == "speaker3"){
			System.out.println("speaker3 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 2, 0.8f, "speaker 3 test");
		} else if(e.controller().name() == "speaker4"){
			System.out.println("speaker4 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 2, 0.8f, "speaker 4 test");
		} else if(e.controller().name() == "speaker5"){
			System.out.println("speaker5 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 4, 0.8f, "speaker 5 test");
		} else if(e.controller().name() == "speaker6"){
			System.out.println("speaker6 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 4, 0.8f, "speaker 6 test");
		} else if(e.controller().name() == "speaker7"){
			System.out.println("speaker7 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 6, 0.8f, "speaker 7 test");
		} else if(e.controller().name() == "speaker8"){
			System.out.println("speaker8 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 6, 0.8f, "speaker 8 test");
		} else if(e.controller().name() == "speaker9"){
			System.out.println("speaker9 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 8, 0.8f, "speaker 9 test");
		} else if(e.controller().name() == "speaker10"){
			System.out.println("speaker10 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 8, 0.8f, "speaker 10 test");
		} else if(e.controller().name() == "speaker11"){
			System.out.println("speaker11 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 10, 0.8f, "speaker 11 test");
		} else if(e.controller().name() == "speaker12"){
			System.out.println("speaker12 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 10, 0.8f, "speaker 12 test");
		} else if(e.controller().name() == "speaker13"){
			System.out.println("speaker13 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 12, 0.8f, "speaker 13 test");
		} else if(e.controller().name() == "speaker14"){
			System.out.println("speaker14 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 12, 0.8f, "speaker 14 test");
		} else if(e.controller().name() == "speaker15"){
			System.out.println("speaker15 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 14, 0.8f, "speaker 15 test");
		} else if(e.controller().name() == "speaker16"){
			System.out.println("speaker16 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 14, 0.8f, "speaker 16 test");
		} else if(e.controller().name() == "speaker17"){
			System.out.println("speaker17 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 16, 0.8f, "speaker 17 test");
		} else if(e.controller().name() == "speaker18"){
			System.out.println("speaker18 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 16, 0.8f, "speaker 18 test");
		} else if(e.controller().name() == "speaker19"){
			System.out.println("speaker19 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 18, 0.8f, "speaker 19 test");
		} else if(e.controller().name() == "speaker20"){
			System.out.println("speaker20 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 18, 0.8f, "speaker 20 test");
		} else if(e.controller().name() == "speaker21"){
			System.out.println("speaker21 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 20, 0.8f, "speaker 21 test");
		} else if(e.controller().name() == "speaker22"){
			System.out.println("speaker22 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 20, 0.8f, "speaker 22 test");
		} else if(e.controller().name() == "speaker23"){
			System.out.println("speaker23 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 22, 0.8f, "speaker 23 test");
		} else if(e.controller().name() == "speaker24"){
			System.out.println("speaker24 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 22, 0.8f, "speaker 24 test");
		} else if(e.controller().name() == "speaker25"){
			System.out.println("speaker25 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 24, 0.8f, "speaker 25 test");
		} else if(e.controller().name() == "speaker26"){
			System.out.println("speaker26 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 24, 0.8f, "speaker 26 test");
		} else if(e.controller().name() == "speaker27"){
			System.out.println("speaker27 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 0, 26, 0.8f, "speaker 27 test");
		} else if(e.controller().name() == "speaker2"){
			System.out.println("speaker28 pressed");
			ConnectionMain.soundController.playSimpleSound("coolmallet_a.wav", 3, 26, 0.8f, "speaker 28 test");
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
