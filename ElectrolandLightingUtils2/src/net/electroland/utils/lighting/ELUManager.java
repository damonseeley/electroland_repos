package net.electroland.utils.lighting;

import java.io.IOException;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Map;
import java.util.Properties;
import java.util.Vector;

import net.electroland.utils.OptionException;
import net.electroland.utils.OptionParser;
import net.electroland.utils.Util;

public class ELUManager implements Runnable {

	private int fps;
	private Vector<Recipient>recipients = new Vector<Recipient>();
	
	public static void main(String args[])
	{
		// test
		try {
			new ELUManager().load();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (OptionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
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
	 * Currently a gross way of doing this.
	 * 
	 */
	public void load() throws IOException, OptionException
	{
		// find lights.properties
		Properties props = new Properties();
		InputStream is = new Util().getClass().getClassLoader().getResourceAsStream("lights.properties");
		if (is != null)
		{
			props.load(is);
		}else{
			throw new OptionException("Please make sure lights.properties is in your classpath.");
		}

		// fps
		try{
			fps = Integer.parseInt(props.getProperty("fps"));			
		}catch(NumberFormatException e){
			fps = 33;
		}

		// parse recipients
		Enumeration <Object> g = props.keys();
		while (g.hasMoreElements())
		{
			String key = ("" + g.nextElement()).trim();
			if (key.toLowerCase().startsWith("recipient."))
			{
				// validate that it has an ID
				int idStart = key.indexOf('.');
				if (idStart == -1 || idStart == key.length() - 1)
				{
					throw new OptionException("no id specified in property " + key);
				}else{
					// get the ID
					String id = key.substring(idStart + 1, key.length());

					// get the props
					Map m = OptionParser.parse("" + props.get(key));

					// load the protocol-appropriate Recipient Class.
					try {
						Recipient r = (Recipient)(new Util().getClass().getClassLoader().loadClass("" + m.get("-class")).newInstance());

						// name, configure, store
						r.setName(id);
						r.configure(m);
						recipients.add(r);
					
					// TODO: friendler error handling here.
					} catch (InstantiationException e) {
						e.printStackTrace();
					} catch (IllegalAccessException e) {
						e.printStackTrace();
					} catch (ClassNotFoundException e) {
						e.printStackTrace();
					}
				}
			}
		}
				
		
		// parse fixtureTypes
		//  for each fixtureType, see if the type has already been defined
		//    * yes? add the detector prototype to the appropriate channel
		//    * no? -create it and store it and add the detector prototype to the appropriate channel

		// parse canvases
		//  for each Canvas
		//    * find the -class
		//    * create an instance of that -class
		//    * run configure()
		//    * case(ELUCanvas2D): 
		//       * get the height and width and store them as dimensions.  if bad, throw OptionException.
		//       * allocate a height x width array
		//    * store the canvas

		// parse fixtures
		//  for each fixture, store the type, tags, recipient, start address
		
		// parse fixture to canvas mappings
		//   for each fixture that is mapped
		//     * find the fixture
		//     * get the fixtureType from the fixture
		//     * find the stored fixtureType
		//     for each prototype detector in the fixtureType
		//      * create a CanvasDetector
		//      * calculate x,y based on the offset store it in the CanvasDetector
		//      * store the width and height
		//      * store the detector model
		//      * calculate which pixels are covered by the boundary, and store them in the CanvasDetector <-- BADNESS IF WE SUPPORT NON-RECTANGLES
		//      * find the recipient and the startChannel from the fixture and the index of the prototype detector in the fixtureType. use those patch the CanvasDetector in as appropriate

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