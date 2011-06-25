package net.electroland.utils.lighting.canvas;

import java.awt.Dimension;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.ELUCanvas;
import net.electroland.utils.lighting.InvalidPixelGrabException;

public class ELUCanvas2D extends ELUCanvas {

	protected Dimension d;	
	protected List<CanvasDetector>detectors = Collections.synchronizedList(new ArrayList<CanvasDetector>());
	
	@Override
	public void configure(Map<String, String> p)
			throws OptionException {
		
		try{
			// get dimensions from config file
			int width = Integer.parseInt(p.get("-width"));
			int height = Integer.parseInt(p.get("-height"));			

			// set dimensions
			d = new Dimension(width,height);
			
			
		}catch(NumberFormatException e){
			throw new OptionException("cannot parse canvas dimensions " + e);
		}		
	}

	@Override
	public void map(CanvasDetector d) throws OptionException {
		detectors.add(d);
		
		// for each index, get the array indices contained within the boundary
		// and store them in the CanvasDetector.

		Rectangle boundary = (Rectangle)d.getBoundary();
		int x1 = boundary.x;
		int y1 = boundary.y;
		int x2 = x1 + boundary.width;
		int y2 = y1 + boundary.height;
		int pixels = boundary.width * boundary.height;
		
		
		for (int y = y1; y <= y2; y++)
		{
			for (int x = x1; x1 <= x2; x++){
				int current = (y * boundary.width) + x;
				 // don't include offscreen pixels
				if (current > 0 && current < pixels){
					d.getIndices().add(current);
				}
			}			
		}
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