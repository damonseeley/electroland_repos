package net.electroland.edmonton.clips;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.util.Map;

import net.electroland.ea.Clip;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class SimpleClip extends Clip {

	private static Logger logger = Logger.getLogger(SimpleClip.class);
	int delay;
	long lastRender;

	@Override
	public void config(ParameterMap primaryParams,
			Map<String, ParameterMap> extendedParams) {

		delay = 1000 / primaryParams.getRequiredInt("fps");

	}

	@Override
	public void init(Map<String, Object> context) {
		// do nothing
		// send a soundcontroller command probably here
	}

	@Override
	public boolean isDone() {
		// play forever, or until someone kills this clip manually
		return false;
	}

	@Override
	public Image getFrame(Image image) {

		//only renders if the delay time (rate) has been exceeded
		// this should work if Clip.image is not cleared during each getFrame, but it appears that it does not
		if (System.currentTimeMillis() - lastRender > delay)
		{
			//Double buffer this to prevent flickering
			BufferedImage bi = new BufferedImage(image.getWidth(null), image.getHeight(null), BufferedImage.TYPE_INT_RGB);
			Graphics g2 = bi.getGraphics();
			g2.setColor(Color.WHITE);
			g2.fillRect(0, 0, 15, 8);

			Graphics g = image.getGraphics();
			g.drawImage(bi, 0, 0, null);
			lastRender = System.currentTimeMillis();
		}

		return image;
	}
}