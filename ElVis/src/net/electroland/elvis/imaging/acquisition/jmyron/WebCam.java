package net.electroland.elvis.imaging.acquisition.jmyron;

import java.awt.image.BufferedImage;
import java.util.Timer;
import java.util.TimerTask;

import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.ImageReceiver;
import JMyron.JMyron;

public class WebCam implements ImageAcquirer {
	JMyron jmyron;
	int width;
	int height;
	float frameRate;
	ImageReceiver imageReceiver;
	Timer timer;

	int imageType = BufferedImage.TYPE_INT_RGB;

	public void setColor(boolean b) {
		if(b) {
			imageType = BufferedImage.TYPE_INT_RGB;
		} else {
			imageType = BufferedImage.TYPE_BYTE_GRAY;			
		}
	}

	
	public WebCam(int w, int h, float fps, ImageReceiver imageReceiver, boolean color) {
		width = w;
		height = h;
		frameRate = fps;
		this.imageReceiver = imageReceiver;
		jmyron = new JMyron();
		jmyron.start(w,h);
		jmyron.findGlobs(0); // turns stuff off I think

		setColor(color);
		timer = new Timer();
	}

	public void start() {
		timer.scheduleAtFixedRate(new JMyronUpdate(),  1, (long)(1000.0f/frameRate));
	}

	public void stopRunning() {
		jmyron.stop();
		timer.cancel();
	}

	public class JMyronUpdate extends TimerTask {

		public void run() {
			jmyron.update();
			BufferedImage bi = new BufferedImage(width, height, imageType);
			bi.setRGB(0,0,width,height,jmyron.image(),0,width); // not sure this works correctly for gray scale
			imageReceiver.addImage(bi);
		}
	}

}
