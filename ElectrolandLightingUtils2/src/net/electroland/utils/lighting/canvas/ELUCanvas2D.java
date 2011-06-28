package net.electroland.utils.lighting.canvas;

import java.awt.Dimension;
import java.awt.Rectangle;
import java.util.Map;

import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.ELUCanvas;
import net.electroland.utils.lighting.InvalidPixelGrabException;

import org.apache.log4j.Logger;

public class ELUCanvas2D extends ELUCanvas {

	private static Logger logger = Logger.getLogger(ELUCanvas2D.class);	

	protected Dimension d;
	int[] lastpixels;

	@Override
	public void configure(Map<String, String> p)
			throws OptionException {
		
		try{
			// get dimensions from config file
			int width = Integer.parseInt(p.get("$width"));
			int height = Integer.parseInt(p.get("$height"));			

			// set dimensions
			d = new Dimension(width,height);
			
			
		}catch(NumberFormatException e){
			throw new OptionException("cannot parse canvas dimensions " + e);
		}		
	}

	@Override
	public void map(CanvasDetector d) throws OptionException {		
		
		// for each index, get the array indices contained within the boundary
		// and store them in the CanvasDetector.

		Rectangle boundary = (Rectangle)d.getBoundary();
		
		int x1 = boundary.x;
		int y1 = boundary.y;
		int x2 = x1 + boundary.width - 1;
		int y2 = y1 + boundary.height - 1;
		int pixels = this.d.width * this.d.height;
		
		for (int y = y1; y <= y2; y++)
		{
			for (int x = x1; x <= x2; x++){
				int current = (y * this.d.width) + x;
				 // don't include offscreen pixels
				if (current > -1 && current < pixels){
					d.getPixelIndices().add(current);
				}
			}			
		}
	}

	@Override
	public CanvasDetector[] sync(int[] pixels) throws InvalidPixelGrabException {
		
		for (CanvasDetector d : detectors)
		{
			d.eval(pixels);
		}
		
		return super.getDetectors();
	}

	public Dimension getDimensions() {
		return d;
	}

	public void debug()
	{
		logger.debug("ELUCanvas2D '" + this.getName() + "' is " + d.width + " by " + d.height + " pixels.");
		for (CanvasDetector cd : detectors)
		{
			logger.debug("ELUCanvas2D '" + this.getName() + "' contains " + cd);
			logger.debug("\tis mapped to pixels " + cd.getPixelIndices());
		}
	}
}