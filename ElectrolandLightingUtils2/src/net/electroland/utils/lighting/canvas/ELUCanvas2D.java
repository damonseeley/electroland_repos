package net.electroland.utils.lighting.canvas;

import java.awt.Dimension;
import java.util.Map;

import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.Detector;
import net.electroland.utils.lighting.ELUCanvas;
import net.electroland.utils.lighting.InvalidPixelGrabException;

public class ELUCanvas2D extends ELUCanvas {

	protected Dimension d;
	
	@Override
	public void configure(Map<String, String> p)
			throws OptionException {
		
		try{
			// get dimensions from config file
			int width = Integer.parseInt(p.get("-width"));
			int height = Integer.parseInt(p.get("-height"));			

			// set dimensions
			d = new Dimension(width,height);

			// TODO: allocate array of pixels (for pixel to detetor mappings)
			
			
		}catch(NumberFormatException e){
			throw new OptionException("cannot parse canvas dimensions " + e);
		}		
	}

	@Override
	public void map(Detector d) throws OptionException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public CanvasDetector[] sync(int[] pixels) throws InvalidPixelGrabException {
		// TODO Auto-generated method stub
		return null;
	}

	public Dimension getDimensions() {
		return d;
	}
}