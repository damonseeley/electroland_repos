package net.electroland.utils.lighting;

import java.awt.Rectangle;
import java.io.IOException;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;

import org.apache.log4j.Logger;

public class ELUManager implements Runnable {

	// TODO: Enumerate object names.
	// TODO: outbourd load() as ELUProperties
	
	private static Logger logger = Logger.getLogger(ELUManager.class);	
		
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
		// Unittest
		try {
			new ELUManager().load();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (OptionException e) {
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

	public void load() throws IOException, OptionException
	{
		load("lights.properties");
	}
	
	/** Configure the system using "lights.properties"
	 * 
	 * Currently a gross way of doing this.
	 * 
	 */
	public void load(String propFileName) throws IOException, OptionException
	{
		ElectrolandProperties ep = new ElectrolandProperties(propFileName);

		// get fps
		fps = ep.getRequiredInt("settings","global","fps");			
		
		// parse recipients
		Iterator <String> recipientNames = ep.getObjectNames("recipient").iterator();
		while (recipientNames.hasNext())
		{
			String name = recipientNames.next();
			
			// TODO: Catch ClassCastException here.
			Recipient r = (Recipient)ep.getRequiredClass("recipient", name, "class");

			// name, configure, store
			r.setName(name);
			r.configure(ep.getAll("recipient", name));
			recipients.put(name, r);
		}
				
		// parse fixtureTypes
		Iterator <String> fixtureTypeNames = ep.getObjectNames("fixtureType").iterator();
		while (fixtureTypeNames.hasNext())
		{
			String name = fixtureTypeNames.next();
			types.put(name, new FixtureType(name, ep.getRequiredInt("fixtureType", name, "channels")));
		}
		
		// patch channels into each fixtureType (e.g., detectors)
		Iterator <String> detectorNames = ep.getObjectNames("detector").iterator();		
		while (detectorNames.hasNext())
		{
			// detector information
			String dname = detectorNames.next();
			int x = ep.getRequiredInt("detector", dname, "x");
			int y = ep.getRequiredInt("detector", dname, "y");
			int width = ep.getRequiredInt("detector", dname, "w");
			int height = ep.getRequiredInt("detector", dname, "h");

			// TODO: Catch ClassCastException here.
			DetectionModel dm = (DetectionModel)ep.getRequiredClass("detector", dname, "model");

			// patch information
			String ftname = ep.getRequired("detector", dname, "fixtureType");
			int index = ep.getRequiredInt("detector", dname, "index");

			// TODO: need to verify that it isn't null
			FixtureType ft = (FixtureType)types.get(ftname);
			ft.detectors.set(index, new Detector(x,y,width,height,dm));
		}

		// parse canvases
		Iterator <String> canvasNames = ep.getObjectNames("canvas").iterator();		
		while (canvasNames.hasNext())
		{
			String canvasName = canvasNames.next();
			ELUCanvas ec = (ELUCanvas)ep.getRequiredClass("canvas", canvasName, "class");
			ec.configure(ep.getAll("canvas", canvasName));
			ec.setName(canvasName);
			canvases.put(canvasName, ec);
		}

		// parse fixtures
		Iterator <String> fixtureNames = ep.getObjectNames("fixture").iterator();		
		while (fixtureNames.hasNext())
		{
			String fixtureName = fixtureNames.next();
			String typeStr = ep.getRequired("fixture", fixtureName, "fixtureType");
			FixtureType type = types.get(typeStr);
			if (type == null){
				throw new OptionException("fixtureType '" + typeStr + "' for object '" + fixtureName + "' of type 'fixture' could not be found.");
			}
			int startAddress = ep.getRequiredInt("fixture", fixtureName, "startAddress");
			String recipStr = ep.getRequired("fixture", fixtureName, "recipient");
			Recipient recipient = recipients.get(recipStr);
			if (recipient == null){
				throw new OptionException("recipient '" + recipStr + "' for object '" + fixtureName + "' of type 'fixture' could not be found.");				
			}
			List<String> tags = ep.getOptionalArray("fixture", fixtureName, "tags");
			Fixture fixture = new Fixture(fixtureName, type, startAddress, recipient, tags);
			
			fixtures.put(fixtureName, fixture);
		}
		
		// parse fixture to canvas mappings (this is the meat of everything)
		Iterator <String> cmapNames = ep.getObjectNames("canvasFixture").iterator();		
		while (fixtureNames.hasNext())
		{
			//   for each fixture to canvas mapping
			String cmapName = cmapNames.next();

			//     * find the fixture
			String fixtureName = ep.getRequired("canvasFixture", cmapName, "fixture");
			Fixture fixture = fixtures.get(fixtureName);
			if (fixture == null){				
				throw new OptionException("fixture '" + fixtureName + "' for object '" + cmapName + "' of type 'canvasFixture' could not be found.");
			}

			//	   * find the canvas
			String cnvsName = ep.getRequired("canvasFixture", cmapName, "canvas");
			ELUCanvas canvas = canvases.get(cnvsName);
			if (canvas == null){				
				throw new OptionException("canvas '" + cnvsName + "' for object '" + cmapName + "' of type 'canvasFixture' could not be found.");
			}
			
			//     for each prototype detector in the fixture (get from FixtureType)
			Iterator<Detector> dtrs = fixture.type.detectors.iterator();
			while (dtrs.hasNext()){
				//      * create a CanvasDetector
				CanvasDetector cd = new CanvasDetector();
				Detector dtr = dtrs.next();

				//      * calculate x,y based on the offset store it in the CanvasDetector
				int offsetX = ep.getRequiredInt("canvasFixture", cmapName, "x");
				int offsetY = ep.getRequiredInt("canvasFixture", cmapName, "y");
				
				Rectangle boundary = new Rectangle(dtr.x + offsetX,
													dtr.y + offsetY,
													dtr.width,
													dtr.height);
				cd.boundary = boundary;
				//      * store the detector model
				cd.detectorModel = dtr.model;

				// map the CanvasDetectors to pixel locations in the pixelgrab
				canvas.map(cd);				
				
				// does it make more sense to do canvas.map(detector, fixture, fixtureType);
				// (same as map.(fixture)) <-- but CanvasDetector should be internally generated

				//	   * find the recipient
				Recipient recipient = fixture.recipient;
				int channel = fixture.startAddress;
				Iterator<Detector> channelDetectors = fixture.type.detectors.iterator();
				while (channelDetectors.hasNext()){
//					recipient.map(channel++, CanvasDetector);					
				}
				
				// recipient.map(fixture);
				//      * find the recipient and the startChannel from the 
				//		  fixture and the index of the prototype detector 
				//		  in the fixtureType. use those patch the CanvasDetector 
				//		  in as appropriate					

			}
		}
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