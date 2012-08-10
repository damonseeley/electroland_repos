package net.electroland.elvis.util;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.image.BufferedImage;
import java.io.IOException;

import net.electroland.elvis.imaging.Unwarp;
import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.ImageReceiver;

import com.googlecode.javacv.CanvasFrame;
import com.googlecode.javacv.FrameGrabber.Exception;
import com.googlecode.javacv.cpp.opencv_core.IplImage;


public class UnwapSettings implements ImageReceiver, KeyListener {
	ElProps props;
	boolean mapIsDirty = false;
	Unwarp uw;
	CanvasFrame canvas;

	public UnwapSettings(ElProps props) throws IOException, Exception {
		this.props = props;
		String cameraStr = props.getProperty("camera", CameraFactory.LOCALAXIS_SRC);
		int w = props.getProperty("srcWidth",640);
		int h = props.getProperty("srcHeight", 480);
		uw = new Unwarp(w,h);
		
		uw.setCenterX(props.getProperty("unwarpCenterX", w*.5));
		uw.setCenterY(props.getProperty("unwarpCenterY", h*.5));
		
		uw.setK1(props.getProperty("unwarpK1", 0));
		uw.setK2(props.getProperty("unwarpK2", 0));
		uw.setP1(props.getProperty("unwarpP1", 0));
		uw.setP2(props.getProperty("unwarpP2", 0));
		mapIsDirty = true;
		
		canvas = new CanvasFrame("Unwarp Settings Util");
		canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE	);        
		canvas.addKeyListener(this);
		canvas.requestFocus(); // not sure why I have to request focus but I do...

		
		ImageAcquirer ir = CameraFactory.camera(cameraStr, w, h, this);
		ir.start();
		
		canvas.requestFocus(); // not sure why I have to request focus but I do...
		
	}
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String arg[]) throws IOException, Exception {
		ElProps props;
		if(arg.length > 0) {
			props = ElProps.init(arg[0]);
		} else {
			props =ElProps.init("blobTracker.props");
		}
		
		new UnwapSettings(props);
		

	}
	@Override
	public void addImage(IplImage i) {
		IplImage dst = i.clone();
		if(mapIsDirty) uw.createMap();
		uw.apply(i, dst);
		canvas.showImage(dst);
	}
	@Override
	public void addImage(BufferedImage i) {
		addImage(IplImage.createFrom(i));
		
	}
	
	@Override
	public void receiveErrorMsg(java.lang.Exception cameraException) {
		System.out.println(cameraException);		
	}
	@Override
	public void keyPressed(KeyEvent arg0) {
		switch(arg0.getKeyCode()) {
		case KeyEvent.VK_A:
			uw.setCenterX(props.inc("unwarpCenterX", -.5, uw.getWidth() * .5));
			System.out.println("unwarpCenterX = " + uw.getCenterX());
			break;
		case KeyEvent.VK_D:
			uw.setCenterX(props.inc("unwarpCenterX", .5, uw.getWidth() * .5));
			System.out.println("unwarpCenterX = " + uw.getCenterX());
			break;
		case KeyEvent.VK_W:
			uw.setCenterY(props.inc("unwarpCenterY", -.5, uw.getHeight() * .5));
			System.out.println("unwarpCenterY = " + uw.getCenterY());
			break;
		case KeyEvent.VK_S:
			uw.setCenterY(props.inc("unwarpCenterY", .5, uw.getHeight() * .5));
			System.out.println("unwarpCenterY = " + uw.getCenterY());
			break;
		case KeyEvent.VK_R:
			uw.setK1(props.inc("unwarpK1", .000001,0));
			System.out.println("unwarpK1 = " + uw.getK1());
			break;
		case KeyEvent.VK_F:
			uw.setK1(props.inc("unwarpK1", -.000001,0));
			System.out.println("unwarpK1 = " + uw.getK1());
			break;
		case KeyEvent.VK_T:
			uw.setK2(props.inc("unwarpK2", .00000001,0));
			System.out.println("unwarpK2 = " + uw.getK2());
			break;
		case KeyEvent.VK_G:
			uw.setK2(props.inc("unwarpK2", -.00000001,0));
			System.out.println("unwarpK2 = " + uw.getK2());
			break;
		case KeyEvent.VK_Y:
			uw.setP1(props.inc("unwarpP1", .00001,0));
			System.out.println("unwarpP1 = " + uw.getP1());
			break;
		case KeyEvent.VK_H:
			uw.setP1(props.inc("unwarpP1", -.00001,0));
			System.out.println("unwarpP1 = " + uw.getP1());
			break;
		case KeyEvent.VK_U:
			uw.setP2(props.inc("unwarpP2", .00001,0));
			System.out.println("unwarpP2 = " + uw.getP2());
			break;
		case KeyEvent.VK_J:
			uw.setP2(props.inc("unwarpP2", -.00001,0));
			System.out.println("unwarpP2 = " + uw.getP2());
			break;
		}
		
	}
	@Override
	public void keyReleased(KeyEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void keyTyped(KeyEvent arg0) {
		// TODO Auto-generated method stub
		
	}	
	
}
