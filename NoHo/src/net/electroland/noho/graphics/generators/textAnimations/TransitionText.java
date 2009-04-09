package net.electroland.noho.graphics.generators.textAnimations;

import java.awt.Color;
import java.awt.Graphics2D;
import java.util.Vector;

import net.electroland.noho.core.LineFormatter;
import net.electroland.noho.core.TextBundle;
import net.electroland.noho.graphics.ImageConsumer;
import net.electroland.noho.graphics.ImageGenerator;
import net.electroland.noho.graphics.Transition;

// this is really a proxy class for the transition
// so most of the normal imagegenerator methods don't do anything (or call the transition)
public class TransitionText extends ImageGenerator {
	
	Transition transition;
	TextBundle textBundle;
	
	protected Vector<String> displayText; // vector of text to display broken into lines
	protected int curLine =0; // line of text in displayText currently rendering

	protected long transitionTime;
	protected long holdTime;
	
	Color color;

	BasicLine basicLine;
	
	public TransitionText(int width, int height, TextBundle textBundle, long transitionTime, long holdTime, Color c, Transition t) {
		super(width, height);
		transition = t;
		this.textBundle = textBundle;
		displayText = LineFormatter.formatString(textBundle.getText());
		setTransitionTime(transitionTime);
		setHoldTime(holdTime);
		color =c;
		preRenderNextLine(c);
	}
	
	public long expectedRunTime() {
		return displayText.size() * (holdTime + transitionTime);
	}
	
	protected void preRenderNextLine(Color c){
		if(curLine < displayText.size()) {
			basicLine = new BasicLine(image.getWidth(), image.getHeight());
			basicLine.setColor(c);
			basicLine.prerenderLine( displayText.get(curLine++));
			basicLine.setDisplayTime(holdTime+transitionTime);
			transition.reset();
			transition.addImage(basicLine);
			transition.startSwitch(transitionTime);
		} else {
			curLine++;
		}
	}
	
	public void setTransitionTime(long transitionTime) {
		this.transitionTime = transitionTime;
		if(this.transitionTime < 0) this.transitionTime = 0;
	}
	public void setHoldTime(long holdTime) {
		this.holdTime = holdTime;
		if(this.holdTime <= 0) this.holdTime =1;
	}

	public void clearBackground(Graphics2D g2d) {
	}
	
	public void drawBackground(Graphics2D g2d) {
	}
	
	public void setBackgroundColor(Color c) {
		transition.setBackgroundColor(c);
	}
	
	
	 protected void render(long dt, long curTime) {
		 // transiton will do all the rendering (and get called in nextframe)
	}
	
	
	/**
	 * Call to render next frame.  Consumer must be set (by calling setConsumer) before calling nextFrame.
	 * @param dt - elapsed time (in milliseconds) since last call to nextFrame
	 */
	public void nextFrame(long dt, long curTime) {
		transition.nextFrame(dt, curTime);
		if(basicLine.isDone()) {
			preRenderNextLine(color);
		}
	}
	
	/**
	 * 
	 * @param consumer - consumer to which generated images should be sent
	 */
	public void setConsumer(ImageConsumer consumer) {
		transition.setConsumer(consumer);
	}
	
	public ImageConsumer getConsumer() {
		return transition.getConsumer();

	}

	public boolean isDone() {
		return curLine > displayText.size();
	}

	public void reset() {
		System.err.println("Can not reset TransitionText");
	}


}