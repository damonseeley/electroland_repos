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
		int x = boundary.x;
		int y = boundary.y;
		int w = boundary.width;
		int h = boundary.height;
		
		// fuck i hate this.
		
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