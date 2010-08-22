package net.electroland.lighting.tools;

import java.awt.Graphics;

import javax.swing.JPanel;

import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Raster;

abstract public class RecipientRepresentation extends JPanel{

	private Recipient recipient;
	private Raster latestFrame;

	public boolean ready = false;

	public RecipientRepresentation(Recipient r)
	{
		this.setRecipient(r);
	}


	// user should overwrite paint() to do something
	// interesting with the raster/recipient
	abstract public void paint(Graphics g);
	
	
	// will be called by AnimationManager
	public void render(Raster r)
	{
		// store this frame so that paint can find it.
		this.latestFrame = r;
		repaint();
	}
	
	public Recipient getRecipient() {
		return recipient;
	}

	public void setRecipient(Recipient recipient) {
		this.recipient = recipient;
		latestFrame = null;
		repaint();
	}

	protected Raster getFrame() {
		return latestFrame;
	}
}