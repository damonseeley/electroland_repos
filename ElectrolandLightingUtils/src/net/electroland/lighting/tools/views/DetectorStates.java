package net.electroland.lighting.tools.views;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.Iterator;

import javax.swing.JPanel;

import net.electroland.lighting.detector.Detector;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.tools.RecipientRepresentation;

public class DetectorStates extends RecipientRepresentation {

	public DetectorStates(Recipient r)
	{
		super(r);
	}
	
	@Override
	protected void paintComponent(Graphics g) {
		g.setColor(Color.BLACK);
		g.fillRect(0, 0, this.getWidth(), this.getHeight());

		// draw raster
		if (this.getRaster() != null)
		{
			BufferedImage raster = (BufferedImage)((this.getRaster().getRaster()));
			g.drawImage(raster, 0, 0, raster.getWidth((JPanel)this), 
										raster.getHeight((JPanel)this), this);
			
		}
		// draw each detector
		Recipient fixture = getRecipient();
		
		Iterator<Detector> i = fixture.getDetectors().iterator();
		while (i.hasNext()){
			Detector d = i.next();
			int rgb = (int)(fixture.getLastEvaluatedValue(d));
			g.setColor(new Color(rgb,rgb,rgb));
			g.fillRect(d.getX(), d.getY(), d.getWidth(), d.getHeight());
			g.setColor(Color.GRAY);
			g.drawRect(d.getX(), d.getY(), d.getWidth(), d.getHeight());
		}

		// draw the border of the fixture
		g.setColor(Color.DARK_GRAY);
		g.drawRect(0, 0, fixture.getPreferredDimensions().width, fixture.getPreferredDimensions().height);					
	}
}