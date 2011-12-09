package net.electroland.edmonton.clips;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.RescaleOp;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import net.electroland.ea.Clip;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class StateToBrightnessImageClip extends Clip {

	private static Logger logger = Logger.getLogger(StateToBrightnessImageClip.class);
	private BufferedImage spriteImage;
	int w,h;
	Map<String, Object> bvals = new HashMap<String, Object>();

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

	public void setBrightValues(Map<String, Object> optionalPostiveDetails){
		bvals = optionalPostiveDetails;
	}

	@Override
	public Image getFrame(Image image) {


		BufferedImage bi = new BufferedImage(image.getWidth(null), image.getHeight(null), BufferedImage.TYPE_INT_RGB);
		Graphics2D g2 = (Graphics2D)bi.getGraphics();
		
		//g2.fillRect(0, 0, w, h);
		for (String key : bvals.keySet()) {
			BrightPoint bp = (BrightPoint) bvals.get(key);
			//need to draw with alpha here...
			if (bp.brightness > 0){
				float fbright = bp.brightness/255.0f;
				float[] scales = { 1f, 1f, 1f, fbright };
				float[] offsets = new float[4];
				RescaleOp rop = new RescaleOp(scales, offsets, null);
				/* Draw the image, applying the filter */
				g2.drawImage(spriteImage,rop,(int)bp.x-5,0);
			}
		}
		g2.dispose();

		Graphics g = image.getGraphics();
		g.drawImage(bi, 0, 0, null);


		return image;
	}
}