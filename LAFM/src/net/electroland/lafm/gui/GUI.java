package net.electroland.lafm.gui;

import java.util.Properties;

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
	private Properties p;
	private int[][] detectors;
	
	public GUI(int width, int height, Conductor conductor, Properties p){
		this.width = width;
		this.height = height;
		this.conductor = conductor;
		this.p = p;
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
		// load detector data for placement on raster
		detectors = new int[25][4];
		for(int i=0; i<25; i++){
			String[] tempdata = ((String)p.get("detector"+i*3)).split(",");
			int[] tempints = {Integer.valueOf(tempdata[0]), Integer.valueOf(tempdata[1]), Integer.valueOf(tempdata[2]), Integer.valueOf(tempdata[3])};
			detectors[i] = tempints;
		}
	}
	
	public void draw(){
		background(0);
		noFill();
		smooth();
		stroke(255);
		pushMatrix();
		translate(10,45);
		drawPattern();
		drawDetectors();
		popMatrix();
	}
	
	public void addActiveShow(ShowThread newShow){
		activeShow = newShow;
		//activeShows.addItem(newShow.getID(), 1);
	}
	
	public void removeActiveShow(String name){
		activeShow = null;
		// TOO MANY ERRORS OCCURRING HERE
		//activeShows.removeItem(name);
	}
	
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
	
	private void drawDetectors(){
		for(int i=0; i<detectors.length; i++){
			//point(detectors[i][0], detectors[i][1]);									// least CPU intensive
			ellipse(detectors[i][0], detectors[i][1], 16, 16);							// most attractive
			//rect(detectors[i][0], detectors[i][1], detectors[i][2], detectors[i][3]);	// most accurate
		}
	}
	
}
