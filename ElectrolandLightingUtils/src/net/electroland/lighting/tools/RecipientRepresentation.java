package net.electroland.lighting.tools;

import java.awt.Graphics;

import javax.swing.JPanel;

import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Raster;

abstract public class RecipientRepresentation extends JPanel{

	private Recipient recipient;
	private Raster raster;

	// crap.  really need a factory here.
	public RecipientRepresentation(Recipient r)
	{
		this.setRecipient(r);
	}
	
	// will be called by AnimationManager
	public void render(Recipient recipient, Raster raster)
	{
		this.setRecipient(recipient);
		this.setRaster(raster);
		repaint();
	}
	
	// user should overwrite paintComponent() to do something
	// interesting with recipient or raster.
	abstract protected void paintComponent(Graphics g);
	
	public Recipient getRecipient() {
		return recipient;
	}

	public void setRecipient(Recipient recipient) {
		this.recipient = recipient;
	}

	public Raster getRaster() {
		return raster;
	}

	public void setRaster(Raster raster) {
		this.raster = raster;
	}
}