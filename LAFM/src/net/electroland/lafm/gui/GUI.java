package net.electroland.lafm.gui;

import java.util.Collection;
import java.util.Iterator;

import net.electroland.detector.Detector;
import net.electroland.lafm.core.Conductor;
import net.electroland.lafm.core.ShowThread;
import processing.core.PApplet;
import promidi.Note;
import controlP5.ControlEvent;
import controlP5.ControlP5;
import controlP5.ScrollList;


public class GUI extends PApplet{

	private static final long serialVersionUID = 1L;
	private int width, height;
	ControlP5 controls;
	ScrollList activeShows;
	private Conductor conductor;
	private ShowThread activeShow;
	private Collection<Detector> detectors;
	
	public GUI(int width, int height, Conductor conductor, Collection<Detector> detectors){
		this.width = width;
		this.height = height;
		this.conductor = conductor;
		this.detectors = detectors;
	}
	
	public void setup(){
		size(width, height);
		controls = new ControlP5(this);
		// setup toggles for testing lights
		for(int i=0; i<24; i++){
			controls.addToggle(str(i+1),false,i*20 + 10,10,15,15).setColorActive(255);
		}
		// setup scrolling list for displaying active shows
		activeShows = controls.addScrollList("activeShows",276,55,120,256);
	}
	
	public void draw(){
		background(0);
		noFill();
		smooth();
		stroke(255);
		pushMatrix();
		translate(10,45);
		drawPattern();
		drawDetectors("lightgroup0");
		popMatrix();
	}
	
	public void addActiveShow(ShowThread newShow){
		activeShow = newShow;
		//activeShows.addItem(newShow.getID(), 1);
	}

	// DON'T DO THIS.  FOR NOW, INSTEAD, LET'S ADD A "SYNC" BUTTON.
	// WHEN YOU PRESS IT, IT POPULATES THE DROPDOWN MENU WITH ALL OF THE SHOWS
	// THAT ARE IN Conductor.getRunningShows().
	
//	public void removeActiveShow(String name){
//		activeShow = null;
//		// TOO MANY ERRORS OCCURRING HERE
//		//activeShows.removeItem(name);
//	}
	
	void controlEvent(ControlEvent e){
		try{
			//String flower = "fixture"+str(Integer.valueOf(e.controller().name())-1);
			
			if(e.controller().parent().name() == "activeShows"){
				
			} else {
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
			}
		}catch(Exception error){
			error.printStackTrace();
		}
	}
	
	private void drawPattern(){
		// the current pattern in play
		if(activeShow != null){
			image(activeShow.getRaster(),0,0,256,256);
		}
		rect(0, 0, 256, 256);
	}
	
	private void drawDetectors(String lightgroup){
		Iterator <Detector> i = detectors.iterator();
		while (i.hasNext()){
			Detector detector = i.next();
			if (detector.getLightGroup().equals(lightgroup))
				ellipse(detector.getX(), detector.getY(), 16, 16);				
		}
	}	
}