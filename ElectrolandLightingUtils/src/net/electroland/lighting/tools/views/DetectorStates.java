package net.electroland.lighting.tools.views;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.event.MouseEvent;
import java.awt.font.FontRenderContext;
import java.awt.font.TextLayout;
import java.awt.image.BufferedImage;
import java.util.Iterator;

import javax.swing.JPanel;
import javax.swing.event.MouseInputListener;

import net.electroland.lighting.detector.DetectionModel;
import net.electroland.lighting.detector.Detector;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.tools.RecipientRepresentation;
import net.electroland.util.Util;
import processing.core.PFont;
import processing.core.PGraphics;
import processing.core.PImage;

public class DetectorStates extends RecipientRepresentation implements MouseInputListener {

	private boolean showDetectors = true;
	private boolean isRunning;
	private String modelName;

	/**
	 * @param r
	 * @param showDetectors - if true, overlay the detectors on the animation frame
	 * @param showModel - if non-null, limit the display to only show detectors of 
	 * 					  the specified type of DetectionModel.
	 */
	public DetectorStates(Recipient r, boolean showDetectors, DetectionModel showModel)
	{
		super(r);
		this.limitDisplayToModel(showModel);
		this.showDetectors = showDetectors;
		this.addMouseListener(this);
	}
	
	public void limitDisplayToModel(DetectionModel model)
	{
		if (model != null){
			modelName = model.getClass().getName();			
		}else{
			showAllModels();
		}
	}

	public void showAllModels()
	{
		modelName = null;
	}

	public void setIsRunning(boolean isRunning)
	{
		this.isRunning = isRunning;
	}
	
	public void setShowDetectors(boolean b)
	{
		this.showDetectors = b;
	}
	
	public void paint(Graphics g) {

		if (isRunning){
			
			g.setColor(Color.LIGHT_GRAY);
			g.fillRect(0, 0, this.getWidth(), this.getHeight());


			if (getFrame() != null){
				if (getFrame().isJava2d()){
					BufferedImage image = (BufferedImage)getFrame().getRaster();
					
					g.drawImage(image, 0, 0, 
							image.getWidth((JPanel)this), 
							image.getHeight((JPanel)this), this);									
				}else{
					PImage image = ((PImage)getFrame().getRaster());
					g.drawImage(image.getImage(), 0, 0, 
							image.width, 
							image.height, this);									
				}
			}
				
			if (showDetectors){
				// draw each detector
				Recipient fixture = getRecipient();
				
				Iterator<Detector> i = fixture.getDetectors().iterator();
				while (i.hasNext()){
					Detector d = i.next();
					// if you get a NullPointerException here, it's probably because there
					// you forgot to patch a channel in your fixture.
					if (modelName == null || modelName.equals(d.getModel().getClass().getName()))
					{
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
		}else{
			Graphics2D g2 = (Graphics2D)g;
			g2.setColor(Color.LIGHT_GRAY);
			g2.fillRect(0, 0, this.getWidth(), this.getHeight());

			g2.setColor(Color.BLACK);
			FontRenderContext frc = g2.getFontRenderContext();
			Font f = new Font("Helvetica",Font.PLAIN, 14);
			String s = new String("System is off.");
			TextLayout tl = new TextLayout(s, f, frc);
			tl.draw(g2, (int)((this.getWidth() - tl.getAdvance()) / 2), (int)(this.getHeight()/2));
		}
	}
	
	public void mouseReleased(MouseEvent arg0) {}

	public void mouseClicked(MouseEvent arg0) {
		
		// TO DO: individual lights on/off mode
		
		// really nice to-do: zoom level.
		
	}

	public void mouseEntered(MouseEvent arg0) {}

	public void mouseExited(MouseEvent arg0) {}

	public void mousePressed(MouseEvent arg0) {}

	public void mouseDragged(MouseEvent arg0) {}

	public void mouseMoved(MouseEvent arg0) {}
}