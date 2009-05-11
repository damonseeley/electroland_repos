package net.electroland.enteractive.gui;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import net.electroland.enteractive.gui.widgets.Button;
import net.electroland.enteractive.gui.widgets.RadioList;
import net.electroland.enteractive.gui.widgets.Slider;
import net.electroland.enteractive.gui.widgets.Widget;
import net.electroland.enteractive.gui.widgets.WidgetEvent;
import net.electroland.enteractive.gui.widgets.WidgetListener;
import net.electroland.lighting.detector.Detector;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Raster;
import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PFont;
import processing.core.PImage;

@SuppressWarnings("serial")
public class GUI extends PApplet implements WidgetListener{
	
	private int width, height;
	private List<Widget> widgets;
	private PFont titlefont;
	private PFont smallfont;
	private Raster raster;
	private Recipient floor, face;
	private int detectorMode = 1;	// 0 disabled, 1 face, 2 floor
	
	public GUI(int width, int height, Recipient floor, Recipient face){
		this.width = width;
		this.height = height;
		this.floor = floor;
		this.face = face;
		widgets = new ArrayList<Widget>();		
		
		for(int i=0; i<3; i++){
			Button b = new Button(this, "button_"+i, 10, i*25+50, 20, 20, 0);
			b.addListener(this);
			widgets.add((Widget)b);
		}
		
		
		RadioList rl = new RadioList(this, "radiolist", 100, 50, 20, 20, 5);
		rl.addListener(this);
		rl.addItem("apple", 0);
		rl.addItem("orange", 1);
		rl.addItem("banana", 2);
		widgets.add((Widget)rl);
		
		for(int i=0; i<3; i++){
			Slider s = new Slider(this, "slider_"+i, 200, i*25+50, 80, 20, 0, 100, (int)(Math.random()*100));
			s.addListener(this);
			widgets.add(s);
		}
		
	}
	
	public void setup(){
		size(width, height);
		frameRate(30);
		noStroke();
		colorMode(PConstants.RGB, 255, 255, 255, 255);
		titlefont = loadFont("depends//fonts//DIN-Medium-18.vlw");
		smallfont = loadFont("depends//fonts//DIN-Regular-10.vlw");
	}
	
	public void draw(){
		background(30);
		noFill();
		stroke(50);
		if(raster.isProcessing()){
			PImage image = (PImage)raster.getRaster();
			image(image, 0, 0);
		}
		drawDetectors();
		//drawTiles();
		/*
		fill(255);
		textFont(titlefont, 18);
		text("ENTERACTIVE CONTROL PANEL", 10, 20);
		textFont(smallfont, 10);
		text("TOGGLES:", 10, 45);
		text("RADIO LIST:", 100, 45);
		text("SLIDERS:", 200, 45);
		Iterator<Widget> i = widgets.iterator();
		while (i.hasNext()){
			i.next().draw();
		}
		*/
	}
	
	public void drawDetectors(){
		stroke(255);
		if(detectorMode == 1){
			try{
				ListIterator<Detector> i = face.getDetectorPatchList().listIterator();
				while(i.hasNext()){
					Detector d = i.next();
					if (d != null){
						//point(d.getX(),d.getY());
						rect(d.getX()-1, d.getY()-1, d.getWidth()+1, d.getHeight()+1);
					}
				}
			} catch(NullPointerException e){
				e.printStackTrace();
			}
		} else if(detectorMode == 2){
			try{
				ListIterator<Detector> i = floor.getDetectorPatchList().listIterator();
				while(i.hasNext()){
					Detector d = i.next();
					if (d != null){
						//point(d.getX(),d.getY());
						rect(d.getX()-1, d.getY()-1, d.getWidth()+1, d.getHeight()+1);
					}
				}
			} catch(NullPointerException e){
				e.printStackTrace();
			}
		}
	}
	
	public void drawTiles(){
		// TODO need to read raster or detector values to determine color
		pushMatrix();
		noFill();
		if(raster != null){
			if(raster.isProcessing()){
				PImage image = (PImage)raster.getRaster();
				for(int y=0; y<image.height; y++){
					for(int x=0; x<image.width; x++){
						int color = image.pixels[(y*image.width) + x];
						stroke(color);
						rect(x*18, y*18, 15, 15);
					}
				}
			}
		}
		popMatrix();
	}
	
	public void setRaster(Raster raster){
		this.raster = raster;
	}
	
	public void setDetectorMode(int detectorMode){
		this.detectorMode = detectorMode;
	}
	
	public void mouseMoved(){
		Iterator<Widget> i = widgets.iterator();
		while (i.hasNext()){
			i.next().mouseMoved(mouseX, mouseY);
		}
	}
	
	public void mouseDragged(){
		Iterator<Widget> i = widgets.iterator();
		while (i.hasNext()){
			i.next().mouseDragged(mouseX, mouseY);
		}
	}
	
	public void mousePressed(){
		Iterator<Widget> i = widgets.iterator();
		while (i.hasNext()){
			i.next().mousePressed(mouseX, mouseY);
		}
	}
	
	public void mouseReleased(){
		Iterator<Widget> i = widgets.iterator();
		while (i.hasNext()){
			i.next().mouseReleased(mouseX, mouseY);
		}
	}

	public void widgetEvent(WidgetEvent we) {
		System.out.println(we.name +" "+ we.widget.getValue());
	}
}
