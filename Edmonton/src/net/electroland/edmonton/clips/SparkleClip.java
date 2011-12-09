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

public class SparkleClip extends Clip {

	private static Logger logger = Logger.getLogger(SparkleClip.class);
	int sparkleWidth;

	@Override
	public void config(ParameterMap primaryParams, Map<String, ParameterMap> extendedParams) {
		sparkleWidth = primaryParams.getRequiredInt("sparkleWidth");
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

		int sparkles = sparkleWidth/6;
		
		BufferedImage bi = new BufferedImage(image.getWidth(null), image.getHeight(null), BufferedImage.TYPE_INT_RGB);
		Graphics g2 = bi.getGraphics();
		
		for (int i=0; i < sparkles; i++){
			int newx = (int)(Math.random()*sparkleWidth);
			int rndClr = (int)(Math.random()*128) + 128;
			g2.setColor(new Color(rndClr,rndClr,rndClr));
			g2.fillRect(newx, 0, sparkles, 16);
		}

		Graphics g = image.getGraphics();
		g.drawImage(bi, 0, 0, null);

		return image;
	}
}