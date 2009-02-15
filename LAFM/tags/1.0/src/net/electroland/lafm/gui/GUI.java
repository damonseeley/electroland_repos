package net.electroland.lafm.gui;

import java.util.Collection;
import java.util.ConcurrentModificationException;
import java.util.Iterator;
import java.util.List;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.detector.Detector;
import net.electroland.lafm.core.Conductor;
import net.electroland.lafm.core.ShowThread;
import processing.core.PApplet;
import processing.core.PImage;
import promidi.Note;
import controlP5.ControlEvent;
import controlP5.ControlP5;
import controlP5.Radio;
import controlP5.ScrollList;


public class GUI extends PApplet{

	private static final long serialVersionUID = 1L;
	private int width, height;
	ControlP5 controls;
	ScrollList sensorShows, rasterList;
	//MultiList multiList;
	private Conductor conductor;
	private int activeShowNum;
	private Collection<Detector> detectors;
	private ShowThread[] showList = new ShowThread[24];
	private boolean thumbsViewable = true;
	private boolean maskRaster = false;
	private boolean viewRaster = true;
	private PImage lightmask0, lightmask1;
	private int testChimeCount = 6;
	
	public GUI(int width, int height, Conductor conductor, Collection<Detector> detectors){
		this.width = width;
		this.height = height;
		this.conductor = conductor;
		this.detectors = detectors;
		activeShowNum = 0;	// default
		this.viewRaster = Boolean.parseBoolean(conductor.systemProps.getProperty("viewRaster"));
		this.thumbsViewable = Boolean.parseBoolean(conductor.systemProps.getProperty("viewThumbs"));
	}
	
	public void setup(){
		size(width, height);
		frameRate(30);
		controls = new ControlP5(this);
		controls.setColorBackground(color(100,100,100,0));
		//controls.setColorForeground(color(255,255,255,10));
		//controls.setColorActive(255);
		lightmask0 = loadImage("depends//images//lightmask0.png");
		lightmask1 = loadImage("depends//images//lightmask1.png");
		
		int xpos = 0;
		int ypos = 0;
		for(int i=0; i<24; i++){	// for each fixture
			if(i+1 != 17 && i+1 != 19){
				// super hacky way to hide the label
				controls.addButton("      "+i, i, xpos*42 + 277, ypos*52 + 11, 31, 31).setColorLabel(0);
				controls.addButton("F" + str(i+1), i, xpos*42 + 276, ypos*52 + 43, 33, 12);
				if(xpos == 4){
					ypos++;
					xpos = 0;
				} else {
					xpos++;
				}
			}
		}		
		
		controls.addTextlabel("sensorlabel","SENSOR PATTERNS:",15,285).setColorValue(0xffff0000);
		Radio r = controls.addRadio("default_sensor_pattern",15,300);
		r.setColorForeground(color(0,54,82,255));
		r.addItem("assigned mode", -1);	// switches back to assigned fixture/show mode
		for(int i=0; i<conductor.sensorShows.length; i++){
			r.addItem(conductor.sensorShows[i], i);
		}
		
		controls.addTextlabel("glocklabel","GLOCKENSPIEL SHOWS:",148,285).setColorValue(0xffff0000);
		controls.addButton("lightgroup test", -1, 148, 298, 114, 12);	// used for calibrating light groups only
		controls.addButton("chimes", -3, 148, 312, 114, 12);			// used for demonstrating chimes only
		for(int i=0; i<conductor.timedShows.length; i++){
			controls.addButton(conductor.timedShows[i], i, 148, 298 + (i+2)*14, 114, 12);//.setColorBackground(color(0,54,82,255));
		}
		
		controls.addTextlabel("settingslabel","SETTINGS:",281,285).setColorValue(0xffff0000);
		controls.addToggle("view_thumbnails", thumbsViewable, 281, 300, 10, 10).setColorForeground(color(0,54,82,255));
		controls.addToggle("view_raster", viewRaster, 370, 300, 10, 10).setColorForeground(color(0,54,82,255));
		controls.addToggle("mask_raster", false, 281, 324, 10, 10).setColorForeground(color(0,54,82,255));
		controls.addToggle("toggle_detectors", true, 370, 324, 10, 10).setColorForeground(color(0,54,82,255));
		controls.addNumberbox("test_chimes",testChimeCount,281,356,50,14);
		controls.addButton("hourly show", -2, 370, 356, 75, 12).setColorBackground(color(0,54,82,255));
		
		controls.addTextlabel("soundlabel","SOUND TESTS:",281,400).setColorValue(0xffff0000);
		controls.addButton("soundtest_1", 1, 281, 413, 100, 12);
		controls.addButton("soundtest_2", 2, 281, 427, 100, 12);
		controls.addButton("soundtest_3", 3, 281, 441, 100, 12);
		controls.addButton("soundtest_global", 4, 281, 455, 100, 12);
	}
	
	public int getChimeCount(){
		return testChimeCount;
	}
	
	public void draw(){
		background(0);
		noFill();
		smooth();
		stroke(255);
		pushMatrix();
		translate(10,10);
		drawPattern();
		if(!maskRaster && viewRaster){
			if(activeShowNum < 14){
				drawDetectors("lightgroup0");
			} else {
				drawDetectors("lightgroup1");
			}
		}
		popMatrix();
		pushMatrix();
		translate(276,10);
		drawThumbs();
		popMatrix();
		rect(10,276,123,260);
		rect(143,276,123,260);
		rect(276,276,200,260);
		//rect(10,276,256,75);
	}
	
	public void drawThumbs(){
		stroke(255);
		noFill();
		int xpos = 0;
		int ypos = 0;
		try{
			showList = new ShowThread[24];
			List <ShowThread> liveShows = conductor.getLiveShows();
			Iterator<ShowThread> i = liveShows.iterator();
			while (i.hasNext()){					// for each active show
				ShowThread s = i.next();
				Collection<DMXLightingFixture> flowers = s.getFlowers();
				Iterator<DMXLightingFixture> f = flowers.iterator();
				while(f.hasNext()){
					DMXLightingFixture flower = f.next();
					showList[Integer.parseInt(flower.getID().split("fixture")[1])-1] = s;
				}
			}
		} catch(ConcurrentModificationException e){
			e.printStackTrace();
		}
		
		for(int n=0; n<24; n++){				// for each fixture
			if(n+1 != 17 && n+1 != 19){
				if(thumbsViewable){
					if(showList[n] != null){
						image(showList[n].getRaster(), xpos*42, ypos*52, 32, 32);
					}
				}
				rect(xpos*42, ypos*52, 32, 32);
				noStroke();
				fill(color(0,54,82,255));
				rect(xpos*42, ypos*52 + 33, 33, 12);
				noFill();
				if(n == activeShowNum){
					stroke(255,0,0);
					rect(xpos*42 - 2, ypos*52 - 2, 36, 48);
				}
				stroke(255);
				/*
				if(thumbsViewable){
					pushMatrix();
					translate(xpos*42, ypos*52);
					if(n < 14){
						drawMiniDetectors("lightgroup0");	// this seems really unnecessary
					} else {
						drawMiniDetectors("lightgroup1");
					}
					popMatrix();
				}
				*/
				if(xpos == 4){
					ypos++;
					xpos = 0;
				} else {
					xpos++;
				}
			}
		}
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
			
			//System.out.println(e.controller().name()  +" "+ e.controller().value());
			if(e.controller().name() == "default_sensor_pattern"){
				// change default sensor show in conductor
				if((int) e.controller().value() < 0){
					conductor.forceSensorShow = false;
				} else {
					conductor.forceSensorShow = true;
					conductor.currentSensorShow = (int) e.controller().value();
				}
			} else if(e.controller().parent().name() == "view_large_raster"){
				//System.out.println(e.controller().value());
				activeShowNum = (int)e.controller().value();
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
			} else if(e.controller().name() == "view_raster"){		// enables/disables raster drawing
				if(e.controller().value() < 1){
					viewRaster = false;
				} else {
					viewRaster = true;
				}
			} else if(e.controller().name() == "toggle_detectors"){	// enables/disables detectors
				List<DMXLightingFixture> fixtures = conductor.getAllFixtures();
				Iterator <DMXLightingFixture> iter = fixtures.iterator();
				while (iter.hasNext()){
					DMXLightingFixture fixture = iter.next();
					fixture.toggleDetectors();
				}
			} else if(e.controller().name().startsWith(" ")){
				activeShowNum = (int)e.controller().value();
			} else if(e.controller().name().startsWith("soundtest")){
				if((int)e.controller().value() == 1){
					conductor.soundManager.playSimpleSound("test1.wav", 1, 1.0f, "test1");
				} else if((int)e.controller().value() == 2){
					conductor.soundManager.playSimpleSound("test2.wav", 2, 1.0f, "test2");
				} else if((int)e.controller().value() == 3){
					conductor.soundManager.playSimpleSound("test6.wav", 6, 1.0f, "test3");
				} else if((int)e.controller().value() == 4){
					int soundID = conductor.soundManager.newSoundID();
					conductor.soundManager.globalSound(soundID,"music.wav",false,1,10000,"globaltest");
				}
			} else if(e.controller().name().startsWith("test_chimes")){
				if((int)e.controller().value() >= 0 && (int)e.controller().value() <= 12)
				testChimeCount = (int)e.controller().value();
			} else {
				// check against list of glockenspiel shows
				boolean glock = false;
				for(int i=0; i<conductor.timedShows.length; i++){
					if(e.controller().name().equals(conductor.timedShows[i])){
						conductor.launchGlockenspiel(i, 6, 0, 0);	// 6 chimes for testing
						glock = true;
					}
				}
				if(e.controller().name().equals("lightgroup test")){
					conductor.launchGlockenspiel(-1, 6, 0, 0);
					glock = true;
				} else if(e.controller().name().equals("chimes")){
					conductor.launchChimes(testChimeCount, 0, 0);	// control chime count from textfield
					glock = true;
				} else if(e.controller().name().equals("hourly show")){	// runs major hourly show + chimes
					conductor.launchGlockenspiel(-3, 6, 0, 0);
					glock = true;
				}
				if(!glock){
					int pitch = (int)e.controller().value() + 36;
					conductor.midiEvent(new Note(pitch, 127, 0));
				}
			}
		}catch(Exception error){
			error.printStackTrace();
		}
	}
	
	private void drawPattern(){
		// the current pattern in play
		if(viewRaster){
			if(showList[activeShowNum] != null){
				image(showList[activeShowNum].getRaster(),0,0,256,256);
			}
			if(maskRaster){
				if(activeShowNum < 13){
					image(lightmask0,0,0,256,256);
				} else {
					image(lightmask1,0,0,256,256);
				}
			}
		}
		rect(0, 0, 256, 256);
	}
	
	private void drawDetectors(String lightgroup){
		int xscale = 256 / conductor.width;
		int yscale = 256 / conductor.height;
		Iterator <Detector> i = detectors.iterator();
		while (i.hasNext()){
			Detector detector = i.next();
			if (detector.getLightGroup().equals(lightgroup))
				ellipse(detector.getX()*xscale, detector.getY()*yscale, 16, 16);				
		}
	}
	
	private void drawMiniDetectors(String lightgroup){
		int xscale = 256 / conductor.width;
		int yscale = 256 / conductor.height;
		Iterator <Detector> i = detectors.iterator();
		while (i.hasNext()){
			Detector detector = i.next();
			if (detector.getLightGroup().equals(lightgroup))
				//ellipse(detector.getX()/8, detector.getY()/8, 3, 3);
				point((detector.getX()*xscale)/8, (detector.getY()*yscale)/8);
		}
	}	
}