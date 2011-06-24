package net.electroland.utils.lighting;

import java.io.IOException;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Map;
import java.util.Vector;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;

public class ELUManager implements Runnable {

	private int fps;
	
	private Hashtable<String, Recipient>recipients 
				= new Hashtable<String, Recipient>();
	private Hashtable<String, FixtureType>types 
				= new Hashtable<String, FixtureType>();
	private Hashtable<String, ELUCanvas>canvases
				= new Hashtable<String, ELUCanvas>();
	private Hashtable<String, Fixture>fixtures
				= new Hashtable<String, Fixture>();
	
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
			
			// TODO: Catch ClassCastException here.
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

			// TODO: Catch ClassCastException here.
			DetectionModel dm = (DetectionModel)ep.getRequiredParamAsClass("detector", dname, "model");

			// patch information
			String ftname = ep.getRequiredParam("detector", dname, "fixtureType");
			int index = ep.getRequiredParamAsInt("detector", dname, "index");

			// TODO: need to verify that it isn't null
			FixtureType ft = (FixtureType)types.get(ftname);
			ft.detectors.set(index, new Detector(x,y,width,height,dm));
		}

		// parse canvases
		Iterator <String> canvasNames = ep.getObjectNames("canvas").iterator();		
		while (canvasNames.hasNext())
		{
			String canvasName = canvasNames.next();
			ELUCanvas ec = (ELUCanvas)ep.getRequiredParamAsClass("canvas", canvasName, "class");
			ec.configure(ep.getParams("canvas", canvasName));
			ec.setName(canvasName);
			canvases.put(canvasName, ec);
			// TODO: allocate a height x width array?
		}

		// parse fixtures
		Iterator <String> fixtureNames = ep.getObjectNames("fixture").iterator();		
		while (fixtureNames.hasNext())
		{
			String fixtureName = fixtureNames.next();
			String typeStr = ep.getRequiredParam("fixture", fixtureName, "fixtureType");
			FixtureType type = types.get(typeStr);
			if (type == null){
				throw new OptionException("fixtureType '" + typeStr + "' for object '" + fixtureName + "' of type 'fixture' could not be found.");
			}
			int startAddress = ep.getRequiredParamAsInt("fixture", fixtureName, "startAddress");
			String recipStr = ep.getRequiredParam("fixture", fixtureName, "recipient");
			Recipient recipient = recipients.get(recipStr);
			if (recipient == null){
				throw new OptionException("recipient '" + recipStr + "' for object '" + fixtureName + "' of type 'fixture' could not be found.");				
			}
			String[] tags = ep.getRequiredParam("fixture", fixtureName, "tags").split(" ");
			Fixture fixture = new Fixture(type, startAddress, recipient, tags);
			fixtures.put(fixtureName, fixture);
			
			// TODO: then patch in the detectors
		}
		
		// parse fixture to canvas mappings
		//   for each fixture that is mapped
		//     * find the fixture
		//     for each prototype detector in the fixture
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
		return canvases.get(name);
	}

	/**
	 * return all canvases (mapped to names)
	 * @return
	 */
	public Map<String, ELUCanvas> getCanvases()
	{
		return canvases;
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
	Vector<Detector> detectors = new Vector<Detector>();
	Recipient recipient;
	
	
	public Fixture(FixtureType type, int startAddress, Recipient recipient, String[] tags){
		this.type = type;
		this.startAddress = startAddress;
		this.tags.addAll(Arrays.asList(tags));
		this.recipient = recipient;
	}
}