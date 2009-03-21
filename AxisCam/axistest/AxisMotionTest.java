package axistest;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;

import axis.AxisCamera;

@SuppressWarnings("serial")

public class AxisMotionTest extends JFrame implements Runnable {
	
	private static AxisCamera ax;
	public BufferedImage image;
	public int w;
	public int h;
	public int offsetx;
	public int offsety;
	public int imageMag;
	
	public MotionDetectorSimple mds;
	
	public AxisMotionTest() { 
		super("AxisMotionTest");

		w=160;
		h=120;
		offsetx = 1;
		offsety = 24;
		imageMag = 4;
		//construct an axis cam thread with baseURL,w,h,compression,color(0,1),user,pass
		ax = new AxisCamera("http://navystreet.dyndns.org:70/",w,h,0,0,"n","n");

		//temp ui stuff
		this.setSize(w*imageMag+offsetx,h*imageMag+offsety);
		this.setVisible(true);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		mds = new MotionDetectorSimple(image);
		
		new Thread(this).start();
		new Thread(ax).run();
		
		image = ax.getImage();
		mds = new MotionDetectorSimple(image);

	}
	
	
	public void paint(Graphics g) {
		
		if (image != null) {
			// draw image avoiding titlebar
			g.drawImage(image, offsetx, offsety, w*imageMag,h*imageMag,this);
		}
	}
	
	public BufferedImage convert(Image im)
	 {
	    BufferedImage bi = new BufferedImage(im.getWidth(null),im.getHeight(null),BufferedImage.TYPE_INT_RGB);
	    Graphics bg = bi.getGraphics();
	    bg.drawImage(im, 0, 0, null);
	    bg.dispose();
	    return bi;
	 }
		
	
	public void run() {
		
		while (true) {
			image = ax.getImage();
			if (image!=null){
				mds.process(image);
			}
			repaint();
		}
		
	}
	
	public static void main(String[] args) {
		AxisMotionTest amt = new AxisMotionTest();
	}
}
