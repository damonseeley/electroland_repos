package net.electroland.lighting.tools.views;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;

import javax.swing.JPanel;

import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.tools.RecipientRepresentation;

@SuppressWarnings("serial")
public class CurrentAnimation extends RecipientRepresentation {

	public CurrentAnimation(Recipient r){
		super(r);
	}
	
	@Override
	public void paint(Graphics g) {

		g.setColor(Color.BLACK);
		g.fillRect(0, 0, this.getWidth(), this.getHeight());

		// draw raster
		if (this.getRaster() != null)
		{
			BufferedImage raster = (BufferedImage)((this.getRaster().getRaster()));
			g.drawImage(raster, 0, 0, raster.getWidth((JPanel)this), 
										raster.getHeight((JPanel)this), this);
			
			// draw the border of the fixture
			Recipient fixture = getRecipient();
			g.setColor(Color.DARK_GRAY);
			g.drawRect(0, 0, fixture.getPreferredDimensions().width, fixture.getPreferredDimensions().height);					
		}
	}
}