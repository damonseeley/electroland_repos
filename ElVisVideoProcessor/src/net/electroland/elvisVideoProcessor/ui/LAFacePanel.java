package net.electroland.elvisVideoProcessor.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.AffineTransform;
import java.awt.image.RenderedImage;

import javax.swing.JPanel;
import javax.swing.Timer;

import net.electroland.elvisVideoProcessor.ElProps;
import net.electroland.elvisVideoProcessor.LAFaceVideoProcessor;


public class LAFacePanel extends JPanel implements ActionListener {

	LAFaceVideoProcessor vidProcessor;
	
	ElProps props = ElProps.THE_PROPS;



	AffineTransform imageScaler;

	String mode = "running";



	public LAFacePanel(LAFaceVideoProcessor vidProcessor) {

		this.vidProcessor = vidProcessor;
		
		Timer t = new Timer( (int) (12.0/1000.0), this);
		t.setInitialDelay(1000);
		t.start();

		setSize(vidProcessor.w, vidProcessor.h);
		setPreferredSize(new Dimension(vidProcessor.w, vidProcessor.h));


	}





	public void setModeString(String s) {
		mode = s;
	}
	public void paint(Graphics g) {
		super.paint(g);
		Graphics2D g2d = (Graphics2D)g;
		
		//g2d.clearRect(0, 0, getWidth(), getHeight());

		RenderedImage ri = vidProcessor.getImage();

		if (ri != null) {
			g2d.drawRenderedImage(ri,  imageScaler);
		}

		renderDrawing(g2d);
		
	}

	public void renderDrawing(Graphics2D g2d) {
		g2d.setColor(Color.RED);
		g2d.drawString(mode, 5, getHeight()-5);
		switch(vidProcessor.getMode()) {
		case crop:
			vidProcessor.crop.renderDrawing(g2d);
			break;
		case setWarp:
			//vidProcessor.getROIConstructor().rescale(vidProcessor.crop.rect);
			vidProcessor.getROIConstructor().renderDrawing(g2d);
			break;
		case setMosiac:
			vidProcessor.getMosaicConstructor().renderDrawing(g2d);
			break;
		}

	}


	public void actionPerformed(ActionEvent e) {
		repaint();				
	}

	public void stop() {
		vidProcessor.stopRunning();
	}


	

}
