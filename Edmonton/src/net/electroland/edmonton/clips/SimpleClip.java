package net.electroland.edmonton.clips;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.TreeSet;
import java.util.Vector;

import javax.imageio.ImageIO;

import net.electroland.ea.Clip;
import net.electroland.utils.OptionException;
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


		if (System.currentTimeMillis() - lastRender > delay)
		{
			Graphics g = image.getGraphics();
			g.setColor(Color.WHITE);
			g.drawRect(0, 0, 10, 10);
			lastRender = System.currentTimeMillis();

		}

		return image;
	}
}