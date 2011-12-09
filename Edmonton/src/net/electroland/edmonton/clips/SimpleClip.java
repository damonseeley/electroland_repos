package net.electroland.edmonton.clips;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.util.Map;

import net.electroland.ea.Clip;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class SimpleClip extends Clip {

	private static Logger logger = Logger.getLogger(SimpleClip.class);
	int w,h;
	long lastRender;

	@Override
	public void config(ParameterMap primaryParams, Map<String, ParameterMap> extendedParams) {

		w = primaryParams.getRequiredInt("width");
		h = primaryParams.getRequiredInt("height");
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

			BufferedImage bi = new BufferedImage(this.baseDimensions.width, 
			        this.getBaseDimensions().height, BufferedImage.TYPE_INT_RGB);
			Graphics g2 = bi.getGraphics();
			g2.setColor(Color.WHITE);
			g2.fillRect(0, 0, w, h);

			Graphics g = image.getGraphics();
			g.drawImage(bi, 0, 0, null);
			lastRender = System.currentTimeMillis();

		return image;
	}
}