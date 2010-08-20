package net.electroland.lighting.tools;

import java.awt.Graphics;

import javax.swing.JPanel;

import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Raster;

abstract public class RecipientRepresentation extends JPanel{

	private Recipient recipient;
	private Raster raster;
	public boolean ready = false;

	public RecipientRepresentation(Recipient r)
	{
		this.setRecipient(r);
	}
	
	// will be called by AnimationManager
	public void render(Raster r)
	{
		this.setRaster(r);
		repaint();
	}
	
	// user should overwrite paint() to do something
	// interesting with the raster/recipient
	abstract public void paint(Graphics g);
	
	public Recipient getRecipient() {
		return recipient;
	}

	public void setRecipient(Recipient recipient) {
		this.recipient = recipient;
		this.setRaster(null);
		repaint();
	}

	public Raster getRaster() {
		return raster;
	}

	public void setRaster(Raster raster) {
		this.raster = raster;
	}
}