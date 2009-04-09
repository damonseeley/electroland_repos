package net.electroland.noho.graphics.generators.textAnimations;

import java.awt.Color;
import java.util.Vector;

import net.electroland.noho.core.LineFormatter;
import net.electroland.noho.core.TextBundle;
import net.electroland.noho.graphics.Switcher;

/**
 * displays a textBundle
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class BasicText extends Switcher {
	Color color;
	TextBundle textBundle;
	
	protected Vector<String> displayText; // vector of text to display broken into lines
	protected int curLine =0; // line of text in displayText currently rendering

	protected long transitionTime;
	protected long holdTime;
	
	BasicLine basicLine;
	public BasicText(int width, int height, TextBundle textBundle, long transitionTime, long holdTime) {
		this(width, height, textBundle, transitionTime, holdTime, Color.WHITE);
	}

	public BasicText(int width, int height, TextBundle textBundle, long transitionTime, long holdTime, Color c) {
		super(width, height);
		this.textBundle = textBundle;
		displayText = LineFormatter.formatString(textBundle.getText());
		setTransitionTime(transitionTime);
		setHoldTime(holdTime);
		color =c;
		preRenderNextLine(c);
	}
	
	//TODO: add camera notification of new line

	protected void preRenderNextLine(Color c){
		if(curLine < displayText.size()) {
			basicLine = new BasicLine(image.getWidth(), image.getHeight());
			basicLine.setColor(c);
			basicLine.prerenderLine( displayText.get(curLine++));
			basicLine.setDisplayTime(holdTime+transitionTime);
			addImage(basicLine);
			startSwitch(transitionTime);
		} else {
			curLine++;
		}
	}
	
	@Override
	public boolean isDone() {
		return curLine > displayText.size();
	}

	@Override
	protected void render(long dt, long curTime) {

		super.render(dt, curTime);
		if(basicLine.isDone()) {
			preRenderNextLine(color);
		}
	}

	public void setHoldTime(long holdTime) {
		this.holdTime = holdTime;
		if(this.holdTime <= 0) this.holdTime =1;
	}

	public void setTransitionTime(long transitionTime) {
		this.transitionTime = transitionTime;
		if(this.transitionTime < 0) this.transitionTime = 0;
	}
	
	

}
