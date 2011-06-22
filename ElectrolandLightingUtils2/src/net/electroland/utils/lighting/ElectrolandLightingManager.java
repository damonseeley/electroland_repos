package net.electroland.utils.lighting;

import java.util.ArrayList;
import java.util.Iterator;

public class ElectrolandLightingManager implements Runnable {

	private int fps;
	private ArrayList<Recipient>recipients;
	
	public void run()
	{
		// TODO: FPS based syncing and calculate measured FPS
	}	

	public void sync()
	{
		Iterator<Recipient> i = recipients.iterator();
		while (i.hasNext()){
			((Recipient)i).sync();
		}
	}
	
	/**
	 * get the FPS that was requested by either lights.properties or overridden
	 * using setTargetFPS.  This is the frame rate the system will do it's best
	 * to achieve.
	 * 
	 * @return
	 */
	public int getTargetFPS()
	{
		return fps;
	}

	/** 
	 * Set the FPS that the system should ATTEMPT to achieve. FPS is translated
	 * to a millisecond delay that is the expected duration of a frame.  During
	 * each frame execution, the time do execute the frame is measured.  If the
	 * time is shorter than the expected duration, the ELU thread will sleep
	 * for the difference.  If the time is longer than the expected duration,
	 * a warning will go out that frame rate has dropped, and immediate 
	 * execution of the next frame will begin.
	 * @param fps
	 */
	public void setTargetFPS(int fps)
	{
		this.fps = fps;
	}
	
	/**
	 * Return the empirically measured frame rate.
	 * @return
	 */
	public int getMeasuredFPS()
	{
		
		return -1;
	}
	
	/**
	 * Turn all channels on all fixtures on.
	 */
	public void allOn()
	{
		
	}

	public void on(String tag)
	{
		
	}
	
	/**
	 * Turn all channels on all fixtures off.
	 */
	public void allOff()
	{
		
	}

	public void off(String tag)
	{
		
	}

	/** Configure the system using "lights.properties"
	 * 
	 */
	public void load()
	{
		// find lights.properties
		// parse recipients (by supported recipient type, alphabetically)
		// parse fixtureTypes
		// parse fixtures
		// parse canvases
		// parse fixture to canvas mappings
		//   for each fixture, 
	}

	/** 
	 * Start autosyncing.  Will sync the latest array sent to each canvas with
	 * the real world lights.
	 */
	public void start()
	{
	}

	/**
	 * stop autosync
	 */
	public void stop()
	{
		
	}
		
	/**
	 * return the canvas referenced by name in lights.properties.
	 * @param name
	 * @return
	 */
	public ELUCanvas getCanvas(String name)
	{
		return null;
	}

	/**
	 * return all canvases.
	 * @return
	 */
	public ELUCanvas[] getCanvases()
	{
		return null;
	}
}