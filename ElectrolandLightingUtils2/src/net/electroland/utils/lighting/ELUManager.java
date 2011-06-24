package net.electroland.utils.lighting;

import java.io.IOException;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Vector;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.Util;

public class ELUManager implements Runnable {

	private int fps;
	
	private Hashtable<String, Recipient>recipients 
				= new Hashtable<String, Recipient>();
	private Hashtable<String, FixtureType>types 
				= new Hashtable<String, FixtureType>();
	
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
		Enumeration<Recipient> i = recipients.elements();
		while (i.hasMoreElements()){
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
		ElectrolandProperties ep = new ElectrolandProperties("lights.properties");

		// get fps
		fps = ep.getRequiredParamAsInt("settings","global","fps");			
		
		// parse recipients
		Iterator <String> recipientNames = ep.getObjectNames("recipient").iterator();
		while (recipientNames.hasNext())
		{
			String name = recipientNames.next();			
			Recipient r = (Recipient)ep.getRequiredParamAsClass("recipient", name, "class");

			// name, configure, store
			r.setName(name);
			r.configure(ep.getParams("recipient", name));
			recipients.put(name, r);
		}
				
		// parse fixtureTypes
		Iterator <String> fixtureTypeNames = ep.getObjectNames("fixtureType").iterator();
		while (fixtureTypeNames.hasNext())
		{
			String name = fixtureTypeNames.next();
			types.put(name, new FixtureType(name, ep.getRequiredParamAsInt("fixtureType", name, "channels")));
		}
		
		// patch channels into each fixtureType
		Iterator <String> detectorNames = ep.getObjectNames("detector").iterator();		
		while (detectorNames.hasNext())
		{
			// detector information
			String dname = detectorNames.next();
			int x = ep.getRequiredParamAsInt("detector", dname, "x");
			int y = ep.getRequiredParamAsInt("detector", dname, "y");
			int width = ep.getRequiredParamAsInt("detector", dname, "w");
			int height = ep.getRequiredParamAsInt("detector", dname, "h");
			DetectionModel dm = (DetectionModel)ep.getRequiredParamAsClass("detector", dname, "model");

			// patch information
			String ftname = ep.getRequiredParam("detector", dname, "fixtureType");
			int index = ep.getRequiredParamAsInt("detector", dname, "index");

			// TODO: need to verify that it isn't null
			FixtureType ft = (FixtureType)types.get(ftname);
			ft.detectors.set(index, new Detector(x,y,width,height,dm));
		}


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

class FixtureType
{
	protected String name;
	protected Vector<Detector> detectors;
	
	public FixtureType(String name, int channels)
	{
		this.name = name;
		detectors = new Vector<Detector>();
		detectors.setSize(channels);
	}
}

class Fixture
{
	FixtureType type;
	int startAddress;
	Vector<String> tags = new Vector<String>();
	Recipient recipient;
	//fixture.f1 = -type PhilipsLEDBar -startAddress 0 -tags "mac:00:00:00 f1" -recipient datagate1
}