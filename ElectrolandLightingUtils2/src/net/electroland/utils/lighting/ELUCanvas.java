package net.electroland.utils.lighting;

import java.util.concurrent.CopyOnWriteArrayList;


abstract public class ELUCanvas {

	private String name;
	protected CopyOnWriteArrayList<CanvasDetector>detectors;
	
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
	
	/**
	 * Get all detectors attached to this detector
	 * @return
	 */
	public CanvasDetector[] getDetectors()
	{
		// TODO: Implement
		return null;
	}
	
	/** Turn all channels attached to this Canvas on.
	 * 
	 */
	public void allOn()
	{		
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
	
	/**
	 * For DIAGNOSTIC USE ONLY, you can set the state of any detector to either
	 * ON or OFF.
	 * 
	 * @param d
	 * @return
	 */
	public void sync(CanvasDetector[] d){
		// TODO: Implement
	}
}