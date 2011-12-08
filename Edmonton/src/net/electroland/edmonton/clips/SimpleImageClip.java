package net.electroland.edmonton.clips;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.TreeSet;
import java.util.Vector;

import javax.imageio.ImageIO;

import net.electroland.ea.Clip;
import net.electroland.ea.clips.AlphanumComparator;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class SimpleImageClip extends Clip {

	private static Logger logger = Logger.getLogger(SimpleImageClip.class);
	private Image spriteImage;
	int delay;
	int w,h;
	long lastRender;

	@Override
	public void config(ParameterMap primaryParams, Map<String, ParameterMap> extendedParams) {

		w = primaryParams.getRequiredInt("width");
		h = primaryParams.getRequiredInt("height");

		String imageFile = primaryParams.getRequired("file");

		try {
			spriteImage = ImageIO.read(new File(imageFile));
		} catch (IOException e) {
			logger.info(this + " could not load image file " + imageFile);
			//return;
			//throw new OptionException(e);
			
		}

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


		if (spriteImage != null){
			Graphics g = image.getGraphics();
			g.drawImage(spriteImage,0,0,this.getBaseDimensions().width,this.getBaseDimensions().height,null);
			g.dispose();
		}


		/*
		BufferedImage bi = new BufferedImage(image.getWidth(null), image.getHeight(null), BufferedImage.TYPE_INT_RGB);
		Graphics g2 = bi.getGraphics();
		g2.setColor(Color.WHITE);
		g2.fillRect(0, 0, w, h);

		Graphics g = image.getGraphics();
		g.drawImage(bi, 0, 0, null);
		lastRender = System.currentTimeMillis();
		 */


		return image;
	}
}