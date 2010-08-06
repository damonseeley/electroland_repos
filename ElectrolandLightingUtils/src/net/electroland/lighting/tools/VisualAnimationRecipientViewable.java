package net.electroland.lighting.tools;

import javax.swing.JPanel;

import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Animation;

// make this a class with a few abstract methods.
abstract public class VisualAnimationRecipientViewable {

	abstract public JPanel getPanel();
	abstract public Recipient getRecipient();
	abstract public Animation getAnimation();

	public Recipient revertPoint;
	
	// these should not be abstract
	final public void allOff()
	{
		
	}
	final public void allOn()
	{
		
	}
	final public void killShow()
	{
		
	}
	final public void skipShow()
	{
		
	}
	// where to put "update" & "revert?"
}
