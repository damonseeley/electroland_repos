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

public class StateToBrightnessClip extends Clip {

	private static Logger logger = Logger.getLogger(StateToBrightnessClip.class);
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

		//only renders if the delay time (rate) has been exceeded
		// this should work if Clip.image is not cleared during each getFrame, but it appears that it does not
//		if (System.currentTimeMillis() - lastRender > delay)
//		{
	        // BELOW IS UNNECESSARY. you could just do:
	        // Graphics g = image.getGraphics();
	        // g.setColor(Color.WHITE);
	        // g.fillRect(0, 0, 15,8);
        //Double buffer this to prevent flickering // BRADLEY: THIS IS UNNECESARRY.
			BufferedImage bi = new BufferedImage(image.getWidth(null), image.getHeight(null), BufferedImage.TYPE_INT_RGB);
			Graphics g2 = bi.getGraphics();
			g2.setColor(Color.WHITE);
			g2.fillRect(0, 0, w, h);

			Graphics g = image.getGraphics();
			g.drawImage(bi, 0, 0, null);
			lastRender = System.currentTimeMillis();

//		}else{
	        // THEN NOTHING HAPPENS HERE.  E.g., nothing is painted so the
		    // image is returned same as it was passed into getFrame, which is
		    // to say blank.
		    // That's causing the flicker.
//		}

		return image;
	}
}