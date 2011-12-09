package net.electroland.edmonton.clips;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Map;

import net.electroland.ea.Clip;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class StateToBrightnessClip extends Clip {

	private static Logger logger = Logger.getLogger(StateToBrightnessClip.class);
	int w,h;
	Map<String, Object> bvals = new HashMap<String, Object>();

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

	public void setBrightValues(Map<String, Object> optionalPostiveDetails){
		bvals = optionalPostiveDetails;
	}

	@Override
	public Image getFrame(Image image) {


		BufferedImage bi = new BufferedImage(image.getWidth(null), image.getHeight(null), BufferedImage.TYPE_INT_RGB);
		Graphics g2 = bi.getGraphics();

		//g2.fillRect(0, 0, w, h);
		for (String key : bvals.keySet()) {
			//logger.info(key);
			BrightPoint bp = (BrightPoint) bvals.get(key);

			//logger.info(bp.brightness);
			//put a filled rect in each x loc
			//need to draw with alpha here...
			if (bp.brightness > 0){
				g2.setColor(new Color(bp.brightness, bp.brightness, bp.brightness));
				g2.fillRect((int)bp.x-6, 0, 11, 16);
			}
		}

		Graphics g = image.getGraphics();
		g.drawImage(bi, 0, 0, null);


		return image;
	}
}