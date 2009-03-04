package net.electroland.enteractive.gui.widgets;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import processing.core.PApplet;

/**
 * This class uses a list of buttons as a "radio button" style list.
 * @author asiegel
 */

public class RadioList extends Widget implements WidgetListener{
	
	private int buttonWidth, buttonHeight, space;
	private List<Widget> items;	// button items

	public RadioList(PApplet canvas, String name, int x, int y, int width, int height, int space) {
		super(canvas, name, x, y, width, 0, 0);		// width and height applied to buttons, not list
		this.buttonWidth = width;
		this.buttonHeight = height;
		this.space = space;
		this.items = new ArrayList<Widget>();
	}
	
	public void addItem(String name, float val){
		Button b = new Button(canvas, name, 0, (buttonHeight+space)*items.size(), buttonWidth, buttonHeight, val);
		b.addListener(this);
		if(items.size() == 0){
			b.silentOn();
		}
		items.add(b); // add item to list
		super.height = ((buttonHeight+space)*items.size())-space;// must adjust the super height for mouse detection
	}

	@Override
	public void draw() {
		// draws the buttons
		canvas.pushMatrix();
		canvas.translate(x, y);
		Iterator<Widget> i = items.iterator();
		while (i.hasNext()){
			i.next().draw();
		}
		canvas.popMatrix();
	}

	@Override
	public void pressed() {
		Iterator<Widget> i = items.iterator();
		while (i.hasNext()){
			i.next().mousePressed(canvas.mouseX-x, canvas.mouseY-y);
		}
	}

	@Override
	public void released() {
	}

	@Override
	public void rollOut() {
		//System.out.println("radio out");
		
	}

	@Override
	public void rollOver() {
		//System.out.println("radio over");
	}

	@Override
	public void cursorMovement() {
		// must report mouse at all times, not just when cursor enters/exits radio button area
		Iterator<Widget> i = items.iterator();
		while (i.hasNext()){
			i.next().mouseMoved(canvas.mouseX-x, canvas.mouseY-y);
		}
	}
	
	public void widgetEvent(WidgetEvent we) {
		if(we.widget.value == value){
			((Button)we.widget).silentOn();
		} else {
			value = we.widget.value;
			// silently deactivate other buttons
			Iterator<Widget> i = items.iterator();
			while (i.hasNext()){
				Widget b = i.next();
				if(!b.name.equals(we.name)){
					((Button)b).silentOff();
				}
			}
			// return active button result to listeners
			WidgetEvent newwe = new WidgetEvent(this, Widget.Constants.PRESSED, true);
			super.newEvent(newwe);
		}
	}

	@Override
	public void dragged() {
		// TODO Auto-generated method stub
		
	}

}
