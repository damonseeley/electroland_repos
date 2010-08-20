package net.electroland.lighting.tools.views;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.util.Iterator;

import javax.swing.JPanel;
import javax.swing.event.MouseInputListener;

import net.electroland.lighting.detector.Detector;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.tools.RecipientRepresentation;
import net.electroland.util.Util;

@SuppressWarnings("serial")
public class DetectorStates extends RecipientRepresentation implements MouseInputListener {

	boolean showDetectors = true;

	public DetectorStates(Recipient r)
	{
		super(r);
		this.addMouseListener(this);
	}
	
	@Override
	public void paint(Graphics g) {

		g.setColor(Color.LIGHT_GRAY);
		g.fillRect(0, 0, this.getWidth(), this.getHeight());


			if (getRaster() != null){
				// seriously: rename getRaster() in the parent.
				BufferedImage image = (BufferedImage)getRaster().getRaster();
				g.drawImage(image, 0, 0, 
						image.getWidth((JPanel)this), 
						image.getHeight((JPanel)this), this);				
			}
			
		if (showDetectors){
			// draw each detector
			Recipient fixture = getRecipient();
			
			Iterator<Detector> i = fixture.getDetectors().iterator();
			while (i.hasNext()){
				Detector d = i.next();
				Byte b = fixture.getLastEvaluatedValue(d);
				if (b != null){
					int rgb = Util.unsignedByteToInt(fixture.getLastEvaluatedValue(d));
					g.setColor(new Color(rgb,rgb,rgb));
					g.fillRect(d.getX(), d.getY(), d.getWidth(), d.getHeight());					
					g.setColor(Color.GRAY);
					g.drawRect(d.getX(), d.getY(), d.getWidth(), d.getHeight());					
				}
			}
		}
	}


	@Override
	public void mouseReleased(MouseEvent arg0) {
		showDetectors = !showDetectors;
	}

	@Override
	public void mouseClicked(MouseEvent arg0) {}

	@Override
	public void mouseEntered(MouseEvent arg0) {}

	@Override
	public void mouseExited(MouseEvent arg0) {}

	@Override
	public void mousePressed(MouseEvent arg0) {}

	@Override
	public void mouseDragged(MouseEvent arg0) {}

	@Override
	public void mouseMoved(MouseEvent arg0) {}
}