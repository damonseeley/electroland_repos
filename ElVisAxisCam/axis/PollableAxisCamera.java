package axis;
/*
 * AxisCamera.java
 *
 * Created on March 12, 2004, 2:53 PM
 */

//obtained from URL: http://forum.java.sun.com/thread.jspa?threadID=494920&start=0&tstart=0

import java.net.*;
import com.sun.image.codec.jpeg.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;

import javax.swing.*;

/**
 *
 * @author David E. Mireles, Ph.D.
 * 
 * Modifications by Damon Seeley to use BufferedImage, Axis 211 compat, URL authorization
 */

public class PollableAxisCamera implements ImageReceiver, ImageAcquirer{
	AxisCamera axisCamera;
	BufferedImage image;

	public PollableAxisCamera (String url, int w, int h, int comp, int color, String user, String pass) {
		axisCamera = new AxisCamera(url,w,h,comp,color,user,pass,this);
	}

	public void addImage(BufferedImage i) {
		image = i;
	}

	public void receiveErrorMsg(Exception cameraException) {
		System.err.println(cameraException);
	}

	public void start() {
		axisCamera.start();
	}

	public void stopRunning() {
		axisCamera.stopRunning();
	}
	public BufferedImage getImage() {
		return image;
	}
}