package net.electroland.lafm.gui;

import net.electroland.lafm.core.Conductor;
import processing.core.PApplet;
import promidi.Note;
import controlP5.ControlEvent;
import controlP5.ControlP5;


public class GUI extends PApplet{

	private static final long serialVersionUID = 1L;
	private int width, height;
	ControlP5 controls;
	private Conductor conductor;
	
	public GUI(int width, int height, Conductor conductor){
		this.width = width;
		this.height = height;
		this.conductor = conductor;
	}
	
	public void setup(){
		size(width, height);
		controls = new ControlP5(this);
		for(int i=0; i<24; i++){
			controls.addToggle(str(i+1),false,i*15 + 10,10,10,10).setColorActive(255);
		}
	}
	
	public void draw(){
		background(0);
		noFill();
		smooth();
		stroke(255);
		drawPattern();
	}
	
	void controlEvent(ControlEvent e){
		try{
			//String flower = "fixture"+str(Integer.valueOf(e.controller().name())-1);
			
			// hack.  should do reverse lookup.
			int pitch = Integer.valueOf(e.controller().name()) + 35;
						
			if(e.value() == 0){ 	// turn fixture off

				conductor.midiEvent(new Note(pitch, 0, 0));
				
				// hook this code up to tell conductor to start a thread.
				
				
//				PGraphics raster = this.createGraphics(256,256,P2D);
//				raster.background(-16777216);
//				System.out.println("off");
//				//Conductor.flowers[flower].sync((PImage)raster);
//				System.out.println(flower);
//				String[] fixtures = Conductor.detectorMngr.getFixtureIds();
//				boolean exists = false;
//				for(int i=0; i<fixtures.length; i++){
//					if(fixtures[i].equals(flower)){
//						exists = true;
//						conductor.detectorMngr.getFixture(flower).sync((PImage)raster);
//						break;
//					}
//				}
				
			} else {				// turn fixture on

				conductor.midiEvent(new Note(pitch, 127, 0));
				
				//Conductor.makeShow(new DiagnosticThread(Conductor.flowers[flower], Conductor.soundManager, 30, 30, this.createGraphics(256,256,"P2D")));
//				PGraphics raster = this.createGraphics(256,256,P2D);
//				raster.background(-1);
//				System.out.println("on");
//				//Conductor.flowers[flower].sync((PImage)raster);
//				System.out.println(flower);
//				String[] fixtures = Conductor.detectorMngr.getFixtureIds();
//				boolean exists = false;
//				for(int i=0; i<fixtures.length; i++){
//					if(fixtures[i].equals(flower)){
//						exists = true;
//						conductor.detectorMngr.getFixture(flower).sync((PImage)raster);
//						break;
//					}
//				}
			}
		}catch(Exception error){
			error.printStackTrace();
		}
	}
	
	private void drawPattern(){
		// the current pattern in play
		rect(10, 40, 150, 150);
	}
	
}
