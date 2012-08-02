package net.electroland.elvis.imaging;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.geom.AffineTransform;

import javax.swing.JFrame;
import javax.swing.Timer;

import net.electroland.elvis.imaging.acquisition.jmyron.WebCam;

public class ProcessorTest  extends JFrame implements KeyListener, ActionListener {
	public static final int scale = 2;
//	public static final int w = 320;
//	public static final int h = 240;

	public static final int w = 160;
	public static final int h = 120;

	AffineTransform scaler ;
	
	PresenceDetector processor;
	
	boolean recalcExtreema = false;


	public ProcessorTest() {
		super();
		processor = new PresenceDetector(w,h);
		processor.start();
		new WebCam(w,h,6.0f,processor, false).start();
//		new NavyCam(processor, true).start();
		setSize(w*scale,h*scale);
		scaler =  new AffineTransform();
		scaler.scale(scale,scale);
		Timer t = new Timer( (int) (12.0/1000.0), this);
		t.setInitialDelay(1000);
		t.start();
		addKeyListener(this);
		setVisible(true);
	}
	
	public void paint(Graphics g) { //used to set the image on the panel
		Graphics2D g2d =(Graphics2D)g;
 		g2d.drawRenderedImage(processor.getBufferedImage(), scaler);
 		if(recalcExtreema) {
 			System.out.println("piel (min,max) = " + processor.getMin() + ", " + processor.getMax());
 			recalcExtreema = false;
 		}
	}



	public void actionPerformed(ActionEvent e) {
		repaint();		
	}

	
	
	
	public void keyPressed(KeyEvent e) {
		switch(e.getKeyCode()) {
		case KeyEvent.VK_F:
			System.out.println(processor.getFPS());
			break;
		case KeyEvent.VK_E:
			processor.recalcExtreema();
			recalcExtreema= true;
			break;
		} 
	}




	public void keyReleased(KeyEvent e) {
	}


	public void keyTyped(KeyEvent e) {
	}

	public static void main(String args[]) {
		new ProcessorTest();
	}

	


}
