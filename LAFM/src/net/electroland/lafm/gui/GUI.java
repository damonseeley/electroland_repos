package net.electroland.lafm.gui;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.detector.Detector;
import net.electroland.lafm.core.Conductor;
import net.electroland.lafm.core.ShowThread;
import processing.core.PApplet;
import processing.core.PGraphics;
import processing.core.PImage;
import promidi.Note;
import controlP5.ControlEvent;
import controlP5.ControlP5;
import controlP5.MultiList;
import controlP5.MultiListButton;
import controlP5.ScrollList;
import controlP5.Toggle;


public class GUI extends PApplet{

	private static final long serialVersionUID = 1L;
	private int width, height;
	ControlP5 controls;
	ScrollList sensorShows, rasterList;
	//MultiList multiList;
	private Conductor conductor;
	private ShowThread activeShow;
	private Collection<Detector> detectors;
	//private PGraphics[] showList = new PGraphics[24];		// all null to start
	private ShowThread[] showList = new ShowThread[24];
	private boolean thumbsViewable = true;
	private boolean maskRaster = false;
	private PImage lightmask0;
	
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
		controls.setColorForeground(color(255,255,255,10));
		controls.setColorActive(color(255,255,255,20));
		lightmask0 = loadImage("depends//images//lightmask0.png");
		
		int xpos = 0;
		int ypos = 0;
		for(int i=0; i<24; i++){	// for each fixture
			if(i+1 != 17 && i+1 != 19){
				controls.addBang(str(i+1),xpos*42 + 276, ypos*52 + 10, 32, 32);
				if(xpos == 4){
					ypos++;
					xpos = 0;
				} else {
					xpos++;
				}
			}
		}
		
		/*
		multiList = controls.addMultiList("multilist",486,10,150,10);
		MultiListButton sensorshowlist = multiList.add("default_sensor_pattern", 1);
		MultiListButton largerasterlist = multiList.add("view_large_raster", 2);
		for(int i=0; i<conductor.sensorShows.length; i++){
			sensorshowlist.add(conductor.sensorShows[i], i);
		}
		for(int i=0; i<24; i++){
			if(i+1 != 17 && i+1 != 19){
				largerasterlist.add(String.valueOf(i), i);
			}
		}
		*/
		
		controls.addToggle("view_thumbnails", true, 370, 220, 10, 10).setColorActive(255);
		controls.addToggle("mask_raster", false, 370, 246, 10, 10).setColorActive(255);
		
		// setup scrolling list for displaying active shows
		sensorShows = controls.addScrollList("default_sensor_pattern",486,20,150,110);
		for(int i=0; i<conductor.sensorShows.length; i++){
			sensorShows.addItem(conductor.sensorShows[i], i);
		}
		
		rasterList = controls.addScrollList("view_large_raster",486, 150, 150, 120);
		for(int i=0; i<24; i++){
			rasterList.addItem(str(i+1), i);
		}
	}
	
	public void draw(){
		background(0);
		noFill();
		smooth();
		stroke(255);
		pushMatrix();
		translate(10,10);
		drawPattern();
		drawDetectors("lightgroup0");
		popMatrix();
		pushMatrix();
		translate(276,10);
		if(thumbsViewable){
			drawRasters();
		}
		popMatrix();
	}
	
	public void drawRasters(){
		stroke(255);
		noFill();
		int xpos = 0;
		int ypos = 0;
		
		//showList = new PGraphics[24];		// all null to start
		showList = new ShowThread[24];
		List <ShowThread> liveShows = conductor.getLiveShows();
		Iterator<ShowThread> i = liveShows.iterator();
		while (i.hasNext()){					// for each active show
			ShowThread s = i.next();
			Collection<DMXLightingFixture> flowers = s.getFlowers();
			Iterator<DMXLightingFixture> f = flowers.iterator();
			while(f.hasNext()){
				DMXLightingFixture flower = f.next();
				//showList[Integer.parseInt(flower.getID().split("fixture")[1])-1] = s.getRaster();
				showList[Integer.parseInt(flower.getID().split("fixture")[1])-1] = s;
			}
		}
		
		for(int n=0; n<24; n++){				// for each fixture
			if(n+1 != 17 && n+1 != 19){
				if(showList[n] != null){
					//image(showList[n], xpos*42, ypos*52, 32, 32);
					image(showList[n].getRaster(), xpos*42, ypos*52, 32, 32);
				}
				rect(xpos*42, ypos*52, 32, 32);
				pushMatrix();
				translate(xpos*42, ypos*52);
				if(n < 17){
					drawMiniDetectors("lightgroup0");	// this seems really unnecessary
				} else {
					drawMiniDetectors("lightgroup1");
				}
				popMatrix();
				if(xpos == 4){
					ypos++;
					xpos = 0;
				} else {
					xpos++;
				}
			}
		}
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
			
			//System.out.println(e.controller().parent().name()  +" "+ e.controller().value());
			if(e.controller().parent().name() == "default_sensor_pattern"){
				// change default sensor show in conductor
				conductor.currentSensorShow = (int) e.controller().value();
			} else if(e.controller().parent().name() == "view_large_raster"){
				System.out.println(e.controller().value());
				if(showList[(int)e.controller().value()] != null){
					activeShow = showList[(int)e.controller().value()];
				} else {
					activeShow = null;
				}
			} else if(e.controller().name() == "view_thumbnails"){	// enables/disables thumbnail raster drawing
				if(e.controller().value() < 1){
					thumbsViewable = false;
				} else {
					thumbsViewable = true;
				}
				//thumbsViewable = Boolean.parseBoolean(String.valueOf(e.controller().value()));
				//System.out.println(e.controller().value());
			} else if(e.controller().name() == "mask_raster"){		// enables/disables masking of large raster
				if(e.controller().value() < 1){
					maskRaster = false;
				} else {
					maskRaster = true;
				}
				//maskRaster = Boolean.parseBoolean(String.valueOf(e.controller().value()));
			} else {
				// hack.  should do reverse lookup.
				int pitch = Integer.valueOf(e.controller().name()) + 35;
				if(e.value() == 0){ 	// turn fixture off
					conductor.midiEvent(new Note(pitch, 0, 0));				
				} else {				// turn fixture on
					conductor.midiEvent(new Note(pitch, 127, 0));
				}
			}
		}catch(Exception error){
			error.printStackTrace();
		}
	}
	
	private void drawPattern(){
		// the current pattern in play
		if(activeShow != null){
			activeShow.getID();
			image(activeShow.getRaster(),0,0,256,256);
			if(maskRaster){
				image(lightmask0,0,0,256,256);
			}
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
	
	private void drawMiniDetectors(String lightgroup){
		Iterator <Detector> i = detectors.iterator();
		while (i.hasNext()){
			Detector detector = i.next();
			if (detector.getLightGroup().equals(lightgroup))
				//ellipse(detector.getX()/8, detector.getY()/8, 3, 3);
				point(detector.getX()/8, detector.getY()/8);
		}
	}	
}