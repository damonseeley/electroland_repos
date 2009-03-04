package net.electroland.enteractive.gui.widgets;

import processing.core.PApplet;

public class Slider extends Widget{

	private float minValue, maxValue, position;
	
	public Slider(PApplet canvas, String name, int x, int y, int width, int height, float minValue, float maxValue, float value) {
		super(canvas, name, x, y, width, height, value);
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.position = (value / (this.maxValue - this.minValue))*width;
	}

	@Override
	public void draw() {
		canvas.pushMatrix();
		canvas.fill(backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3]);
		canvas.rect(x, y, width, height);	// background rectangle
		canvas.fill(activeColor[0], activeColor[1], activeColor[2], activeColor[3]);
		canvas.rect(x, y, position, height);
		canvas.popMatrix();
	}

	@Override
	public void pressed() {
		position = canvas.mouseX-x;
		value = ((position / width) * (this.maxValue - this.minValue)) + this.minValue;
		WidgetEvent we = new WidgetEvent(this, Widget.Constants.PRESSED, true);
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
		if(canvas.mouseX-x < 0){
			position = 0;
			value = minValue;
		} else if(canvas.mouseX-x > width){
			position = width;
			value = maxValue;
		} else {
			position = (canvas.mouseX-x);
			value = ((position / width) * (this.maxValue - this.minValue)) + this.minValue;
		}
		//System.out.println(value);
		WidgetEvent we = new WidgetEvent(this, Widget.Constants.DRAGGED, true);
		super.newEvent(we);
	}

}
