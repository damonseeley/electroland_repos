package net.electroland.utils.lighting;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import net.electroland.utils.OptionException;


abstract public class ELUCanvas {

	private String name;
	protected List<CanvasDetector>detectors = Collections.synchronizedList(new ArrayList<CanvasDetector>());
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}	

	/**
	 * Send an array of pixels to be synced with all fixtures/detectors that
	 * are attached to this canvas.  The return will be a clone of all the 
	 * * detectors in that is in a 0,0 (left,top) coordinate space, with the 
	 * evaluated byte values in each detector.
	 * 
	 * @param a
	 * @return
	 * @throws BadArrayException - will be thrown if the array does not conform
	 * to any constraints defined by the surface that the concrete instance of
	 * this class exposes.  For example, ELUCanvas2D would require that the
	 * length of the array be exactly 
	 * 
	 *   ELUCanvas2D.getWidth() * ELUCanvas2D.getHeight()
	 *   
	 * Alternately, the array may throw an exception if things like bitdepth
	 * are violated within data in the array.
	 * 
	 */
	abstract public CanvasDetector[] sync(int[] pixels) throws InvalidPixelGrabException;

	abstract public void configure(Map<String,String> properties) throws OptionException;

	abstract public void map(CanvasDetector d) throws OptionException;

	/**
	 * print debugging info.
	 */
	abstract public void debug();
	
	/**
	 * Get all detectors attached to this detector
	 * @return
	 */
	public CanvasDetector[] getDetectors()
	{
		return detectors.toArray(new CanvasDetector[detectors.size()]);
	}

	public void addDetector(CanvasDetector cd) throws OptionException
	{
		detectors.add(cd);
		map(cd);
	}
	
	/** Turn all channels attached to this Canvas on.
	 * 
	 */
	public void allOn()
	{
//		for (CanvasDetector d : detectors)
//		{
			// how is this going to work?  allOn is a protocol specific call, 
			// so this needs to go to ARTNetRecipient.
			
			// maybe have a boolean for useSync	 and another for isOn, and
			// just set those for each detector.  then ARTNet can deal.
			
			// recipient can call useSync 
			
//		}
		// TODO: Implement.  This is a wrapper around
		// setting all CanvasDetectors to ON and calling sync.
	}

	/** Turn all channels attached to this Canvas off.
	 * 
	 */
	public void allOff()
	{
		// TODO: Implement
		// setting all CanvasDetectors to OFF and calling sync.
	}
}