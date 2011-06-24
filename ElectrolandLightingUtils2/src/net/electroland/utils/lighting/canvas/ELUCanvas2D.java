package net.electroland.utils.lighting.canvas;

import java.awt.Dimension;
import java.util.Hashtable;

import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.Detector;
import net.electroland.utils.lighting.ELUCanvas;
import net.electroland.utils.lighting.InvalidPixelGrabException;

public class ELUCanvas2D extends ELUCanvas {

	protected Dimension d;
	
	@Override
	public void Configure(Hashtable<String, String> properties)
			throws OptionException {
		// TODO Auto-generated method stub
		
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