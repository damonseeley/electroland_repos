package net.electroland.enteractive.gui.widgets;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import processing.core.PApplet;

/**
 * Abstract class for all GUI widgets.
 * @author asiegel
 */

public abstract class Widget{

	private List<WidgetListener> listeners;
	protected PApplet canvas;
	protected String name;
	protected int x, y, width, height;	// dimensions
	protected float value;				// internal value of widget
	protected boolean mouseOver = false;
	protected boolean mouseDown = false;
	protected int[] backgroundColor = {100, 100, 100, 255};		// standard color
	protected int[] foregroundColor = {255, 255, 255, 255};		// roll-over color
	protected int[] activeColor = {0, 150, 255, 255};				// "on" color
	protected int[] activeForegroundColor = {0, 255, 255, 255};	// "on" with roll-over color
	
	public Widget(PApplet canvas, String name, int x, int y, int width, int height, float value){
		this.canvas = canvas;
		this.name = name;
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
		this.value = value;
		listeners = new ArrayList<WidgetListener>();
	}
	
	final public void addListener(WidgetListener wl){
		listeners.add(wl);
	}
	
	final public void removeListener(WidgetListener wl){
		listeners.remove(wl);
	}
	
	final public void newEvent(WidgetEvent we){
		Iterator<WidgetListener> i = listeners.iterator();
		while (i.hasNext()){
			i.next().widgetEvent(we);
		}
	}
	
	
	
	
	/* MOUSE EVENT FUNCTIONS */
	
	public abstract void draw();
	public abstract void pressed();
	public abstract void released();
	public abstract void dragged();
	public abstract void rollOver();
	public abstract void rollOut();
	public abstract void cursorMovement();
	
	final public void mouseMoved(int mouseX, int mouseY){
		if(mouseInside(mouseX, mouseY)){	// if within constraints, activate rollOver
			if(!mouseOver){
				mouseOver = true;
				rollOver();
			}
		} else {							// if outside constraints, activate rollOut
			if(mouseOver){
				mouseOver = false;
				rollOut();
			}
		}
		cursorMovement();					// verbose movement repeater (needed for embedded items)
	}
	
	final public void mouseDragged(int mouseX, int mouseY){
		if(mouseDown){
			dragged();
		}
	}
	
	final public void mousePressed(int mouseX, int mouseY){
		if(mouseInside(mouseX, mouseY)){
			mouseDown = true;
			pressed();
		}
	}

	final public void mouseReleased(int mouseX, int mouseY){
		if(mouseDown){
			released();
			mouseDown = false;
		}
	}
	
	final private boolean mouseInside(int mouseX, int mouseY){
		if((mouseX >= x && mouseX <= x+width) && (mouseY >= y && mouseY <= y+height)){
			return true;
		} else {
			return false;
		}
	}
	
	final public String getName(){
		return name;
	}
	
	
	
	
	/* COLOR SETTER FUNCTIONS */
	
	final public void setBackgroundColor(int red, int green, int blue, int alpha){
		backgroundColor[0] = red;
		backgroundColor[1] = green;
		backgroundColor[2] = blue;
		backgroundColor[3] = alpha;
	}
	
	final public void setForegroundColor(int red, int green, int blue, int alpha){
		foregroundColor[0] = red;
		foregroundColor[1] = green;
		foregroundColor[2] = blue;
		foregroundColor[3] = alpha;
	}
	
	final public void setActiveColor(int red, int green, int blue, int alpha){
		activeColor[0] = red;
		activeColor[1] = green;
		activeColor[2] = blue;
		activeColor[3] = alpha;
	}
	
	final public void setActiveForegroundColor(int red, int green, int blue, int alpha){
		activeForegroundColor[0] = red;
		activeForegroundColor[1] = green;
		activeForegroundColor[2] = blue;
		activeForegroundColor[3] = alpha;
	}
	
	
	
	
	
	final public float getValue(){
		return value;
	}
	
	
	
	public static class Constants{
		public static int PRESSED = 0;
		public static int RELEASED = 1;
		public static int ROLLOVER = 2;
		public static int ROLLOUT = 3;
		public static int DRAGGED = 4;
	}
	
}
