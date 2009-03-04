package net.electroland.enteractive.gui.widgets;

import processing.core.PApplet;

/**
 * Button has simple on/off behavior.
 * @author asiegel
 */

public class Button extends Widget{
	
	private boolean on = false;
	
	public Button(PApplet canvas, String name, int x, int y, int width, int height, float value){
		super(canvas, name, x, y, width, height, value);
	}
	
	@Override
	public void draw(){
		canvas.pushMatrix();
		canvas.translate(x, y);
		if(mouseOver && on){
			canvas.fill(activeForegroundColor[0], activeForegroundColor[1], activeForegroundColor[2], activeForegroundColor[3]);
		} else if(mouseOver && !on){
			canvas.fill(foregroundColor[0], foregroundColor[1], foregroundColor[2], foregroundColor[3]);
		} else if(on){
			canvas.fill(activeColor[0], activeColor[1], activeColor[2], activeColor[3]);
		} else {
			canvas.fill(backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3]);
		}
		canvas.rect(0,0,width,height);
		canvas.popMatrix();
	}
	
	public void silentToggle(){
		on = !on;
	}
	
	public void silentOff(){
		on = false;
	}
	
	public void silentOn(){
		on = true;
	}

	@Override
	public void pressed() {
		on = !on;
		WidgetEvent we = new WidgetEvent(this, Widget.Constants.PRESSED, on);
		super.newEvent(we);
	}

	@Override
	public void released() {
	}

	@Override
	public void rollOut() {
	}

	@Override
	public void rollOver() {
	}

	@Override
	public void cursorMovement() {
	}

	@Override
	public void dragged() {
		// TODO Auto-generated method stub
		
	}	
	
}
