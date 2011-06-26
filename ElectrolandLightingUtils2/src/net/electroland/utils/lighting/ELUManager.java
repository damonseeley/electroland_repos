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

	final public static String FIXTURE = "fixture"; 
	final public static String FIXTURE_TYPE = "fixtureType"; 
	final public static String FPS = "fps"; 
	final public static String RECIPIENT = "recipient"; 
	final public static String DETECTOR = "detector"; 
	final public static String CANVAS = "canvas"; 
	final public static String CANVAS_FIXTURE = "canvasFixture"; 
	
	private static Logger logger = Logger.getLogger(ELUManager.class);	

	private int fps;
	
	private Hashtable<String, Recipient>recipients 
				= new Hashtable<String, Recipient>();
	private Hashtable<String, ELUCanvas>canvases
				= new Hashtable<String, ELUCanvas>();
	private Hashtable<String, Fixture>fixtures
				= new Hashtable<String, Fixture>();
	private Hashtable<String, FixtureType>types 
				= new Hashtable<String, FixtureType>();
		
	public static void main(String args[])
	{
		// Unit test
		try {
			new ELUManager().load().debug();

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

	public ELUManager load() throws IOException, OptionException
	{
		return load("lights.properties");
	}
	
	/** Configure the system using "lights.properties"
	 * 
	 * Currently a gross way of doing this.
	 * 
	 */
	public ELUManager load(String propFileName) throws IOException, OptionException
	{
		logger.info("ELUManager loading " + propFileName);
		ElectrolandProperties ep = new ElectrolandProperties(propFileName);

		// get fps
		fps = ep.getRequiredInt("settings","global",FPS);			
		
		// parse recipients
		Iterator <String> recipientNames = ep.getObjectNames(RECIPIENT).iterator();
		while (recipientNames.hasNext())
		{
			String name = recipientNames.next();
			
			// TODO: Catch ClassCastException here.
			Recipient r = (Recipient)ep.getRequiredClass(RECIPIENT, name, "class");

			// name, configure, store
			r.setName(name);
			r.configure(ep.getAll(RECIPIENT, name));
			recipients.put(name, r);
		}
				
		// parse fixtureTypes
		Iterator <String> fixtureTypeNames = ep.getObjectNames(FIXTURE_TYPE).iterator();
		while (fixtureTypeNames.hasNext())
		{
			String name = fixtureTypeNames.next();
			types.put(name, new FixtureType(name, ep.getRequiredInt(FIXTURE_TYPE, name, "channels")));
		}
		
		// patch channels into each fixtureType (e.g., detectors)
		Iterator <String> detectorNames = ep.getObjectNames(DETECTOR).iterator();		
		while (detectorNames.hasNext())
		{
			// detector information
			String dname = detectorNames.next();
			int x = ep.getRequiredInt(DETECTOR, dname, "x");
			int y = ep.getRequiredInt(DETECTOR, dname, "y");
			int width = ep.getRequiredInt(DETECTOR, dname, "w");
			int height = ep.getRequiredInt(DETECTOR, dname, "h");

			try{
				DetectionModel dm = (DetectionModel)ep.getRequiredClass(DETECTOR, dname, "model");				

				// patch information
				String ftname = ep.getRequired(DETECTOR, dname, FIXTURE_TYPE);
				int index = ep.getRequiredInt(DETECTOR, dname, "index");

				// TODO: need to verify that it isn't null
				FixtureType ft = (FixtureType)types.get(ftname);
				ft.detectors.set(index, new Detector(x,y,width,height,dm));

			}catch(ClassCastException e)
			{
				// TODO: Proper error message
				throw new OptionException(e);
			}

		}

		// parse canvases
		Iterator <String> canvasNames = ep.getObjectNames(CANVAS).iterator();		
		while (canvasNames.hasNext())
		{
			String canvasName = canvasNames.next();
			ELUCanvas ec = (ELUCanvas)ep.getRequiredClass(CANVAS, canvasName, "class");
			ec.configure(ep.getAll(CANVAS, canvasName));
			ec.setName(canvasName);
			canvases.put(canvasName, ec);
		}

		// parse fixtures
		Iterator <String> fixtureNames = ep.getObjectNames(FIXTURE).iterator();		
		while (fixtureNames.hasNext())
		{
			String fixtureName = fixtureNames.next();
			String typeStr = ep.getRequired(FIXTURE, fixtureName, FIXTURE_TYPE);
			FixtureType type = types.get(typeStr);
			if (type == null){
				throw new OptionException("fixtureType '" + typeStr + "' for object '" + fixtureName + "' of type 'fixture' could not be found.");
			}
			int startAddress = ep.getRequiredInt(FIXTURE, fixtureName, "startAddress");
			String recipStr = ep.getRequired(FIXTURE, fixtureName, RECIPIENT);
			Recipient recipient = recipients.get(recipStr);
			if (recipient == null){
				throw new OptionException("recipient '" + recipStr + "' for object '" + fixtureName + "' of type 'fixture' could not be found.");				
			}
			List<String> tags = ep.getOptionalArray(FIXTURE, fixtureName, "tags");
			Fixture fixture = new Fixture(fixtureName, type, startAddress, recipient, tags);
			
			fixtures.put(fixtureName, fixture);
		}
		
		// parse fixture to canvas mappings (this is the meat of everything)
		Iterator <String> cmapNames = ep.getObjectNames(CANVAS_FIXTURE).iterator();
		while (cmapNames.hasNext())
		{			
			//   for each fixture to canvas mapping
			String cmapName = cmapNames.next();

			//     * find the fixture
			String fixtureName = ep.getRequired(CANVAS_FIXTURE, cmapName, FIXTURE);
			Fixture fixture = fixtures.get(fixtureName);
			if (fixture == null){				
				throw new OptionException("fixture '" + fixtureName + "' for object '" + cmapName + "' of type 'canvasFixture' could not be found.");
			}

			//	   * find the canvas
			String cnvsName = ep.getRequired(CANVAS_FIXTURE, cmapName, CANVAS);
			ELUCanvas canvas = canvases.get(cnvsName);
			if (canvas == null){				
				throw new OptionException("canvas '" + cnvsName + "' for object '" + cmapName + "' of type 'canvasFixture' could not be found.");
			}

			// counter for recipient mappings
			int channel = fixture.startAddress;			

			//	   * find the recipient
			Recipient recipient = fixture.recipient;	

			
			//     for each prototype detector in the fixture (get from FixtureType)
			Iterator<Detector> dtrs = fixture.type.detectors.iterator();
			while (dtrs.hasNext()){
				//      * create a CanvasDetector
				CanvasDetector cd = new CanvasDetector();
				cd.tags = fixture.tags;
				Detector dtr = dtrs.next();

				//      * calculate x,y based on the offset store it in the CanvasDetector
				double offsetX = ep.getRequiredDouble(CANVAS_FIXTURE, cmapName, "x");
				double offsetY = ep.getRequiredDouble(CANVAS_FIXTURE, cmapName, "y");

				double scaleX = ep.getRequiredDouble(CANVAS_FIXTURE, cmapName, "xScale");
				double scaleY = ep.getRequiredDouble(CANVAS_FIXTURE, cmapName, "yScale");				

				Rectangle boundary = new Rectangle((int)((scaleX * (dtr.x + offsetX))),
													(int)((scaleY * (dtr.y + offsetY))),
													(int)(dtr.width * scaleX),
													(int)(dtr.height * scaleY));
				cd.boundary = boundary;
				//      * store the detector model
				cd.detectorModel = dtr.model;

				// map the CanvasDetectors to pixel locations in the pixelgrab
				canvas.addDetector(cd);
				
				// map the CanvasDetector to a channel in the recipient.
				recipient.map(channel++, cd);
			}
		}
		return this;
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

	public void debug()
	{
		logger.debug("FPS set to " + fps);
		
		Enumeration<ELUCanvas> e = canvases.elements();
		while (e.hasMoreElements())
		{
			e.nextElement().debug();
		}

		Enumeration<Recipient> r = recipients.elements();
		while (r.hasMoreElements())
		{
			r.nextElement().debug();
		}				
	}
}