package net.electroland.utils.lighting;

import java.awt.Rectangle;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.FrameRateRingBuffer;
import net.electroland.utils.OptionException;

import org.apache.log4j.Logger;

public class ELUManager implements Runnable {

	// TODO: move into an Enum.
	final public static String FIXTURE = "fixture"; 
	final public static String FIXTURE_TYPE = "fixtureType"; 
	final public static String FPS = "fps"; 
	final public static String RECIPIENT = "recipient"; 
	final public static String DETECTOR = "detector"; 
	final public static String CANVAS = "canvas"; 
	final public static String CANVAS_FIXTURE = "canvasFixture"; 
	final public static String TEST = "test";
	final public static String TEST_SUITE = "testSuite";
	
	private static Logger logger = Logger.getLogger(ELUManager.class);	

	private int fps;
	
	private HashMap<String, Recipient> recipients;
	private HashMap<String, ELUCanvas>canvases;
	private HashMap<String, Test>tests;
	private HashMap<String, TestSuite>suites;

	private Thread thread;
	boolean isRunning = false;
	boolean isRunningTest = false;
	
	// assume a general 45 fps over 10 seconds.
	private FrameRateRingBuffer fpsBuffer = new FrameRateRingBuffer(45 * 10);

	// Unit test.  Does sweep continuously.
	public static void main(String args[])
	{
		try {

			ELUManager elu = new ELUManager();
			boolean isOn = true;

			Map<String,Integer> commands = new HashMap<String,Integer>();
			commands.put("start", 0);
			commands.put("stop", 1);
			commands.put("fps", 2);
			commands.put("allon", 3);
			commands.put("alloff", 4);
			commands.put("list", 5);
			commands.put("load", 6);
			commands.put("test", 7);
			commands.put("quit", 8);
			commands.put("on", 9);
			commands.put("off", 10);
			commands.put("set", 11);

			while(isOn)
			{
				try{
					System.out.print(">");

					java.io.BufferedReader stdin = 
							new java.io.BufferedReader(
									new java.io.InputStreamReader(System.in));

					String input[] = stdin.readLine().split(" ");
					Integer i = commands.get(input[0].toLowerCase());

					if (i == null || input[0] == "?"){
						System.out.println("unknown command " + input[0]);
						System.out.println("--");
						System.out.println("The following commands are valid:");
						System.out.println("\tload [light properties file name]");
						System.out.println("\tlist");
						System.out.println("\tstart");
						System.out.println("\tstop");
						System.out.println("\tfps");
						System.out.println("\tfps [desired fps]");						
						System.out.println("\tallon");
						System.out.println("\talloff");
						System.out.println("\ton [tag]");
						System.out.println("\toff [tag]");
						System.out.println("\tset [tag] [value 0-255]");
						System.out.println("\ttest [testSuite]");
						System.out.println("\tquit");
					}else{
						switch(i.intValue()){
						case(0):
							elu.start();
							break;
						case(1):
							elu.stop();
							break;
						case(2):
							if (input.length == 1)
								System.out.println("Current measured fps = " + elu.getMeasuredFPS());
							else{
								try{
									int fps = Integer.parseInt(input[1]);
									if (fps > 0)
										elu.fps = fps;
									else
										System.out.println("Illegal fps: " + input[1]);
								}catch(NumberFormatException e)
								{
									System.out.println("Illegal fps: " + input[1]);
								}
							}
							break;
						case(3):
							elu.allOn();
							break;
						case(4):
							elu.allOff();
							break;
						case(5):
							elu.debug();
							break;
						case(6):
							if (input.length == 1)
								elu.load("lights.properties");
							else
								elu.load(input[1]);
							break;
						case(7):
								if (input.length == 2)
									elu.runTest(input[1]);
								else
									System.out.println("usage: run [testSuite]");
							break;
						case(8):
							elu.stop();
							isOn = false;
							break;
						case(9):
							elu.on(input[1]);
							break;
						case(10):
							elu.off(input[1]);
							break;
						case(11):
							if (input.length == 3)
							{
								try{
									int val = Integer.parseInt(input[2]);
									if (val > -1 && val < 256)
										elu.set(input[1],(byte)val);
									else
										System.out.println("usage: set [tag] [value(0-255)]");
								}catch(NumberFormatException e)
								{
									System.out.println("usage: set [tag] [value(0-255)]");
								}
							}else
							{
								System.out.println("usage: set [tag] [value(0-255)]");
							}
							break;
						}
					}
				}catch (java.io.IOException e){
					logger.error(e);
				}			
			}
			
		} catch (OptionException e) {
			logger.error(e);
		}
		
	}
	
	public final void run()
	{
		long targetDelay = (int)(1000.0 / fps);

		while (isRunning)
		{
			// record start time
			long start = System.currentTimeMillis();

			// sync all canvases to recipients
			this.syncAllLights();

			// how long did it take to execute?
			long duration = System.currentTimeMillis() - start;
			
			long delay = duration > targetDelay ? 0 : targetDelay - duration;

			try {
				Thread.sleep(delay);
			} catch (InterruptedException e) {
				logger.error(e);
			}
		}
		thread = null;
	}	

	/** 
	 * Start autosyncing.  Will sync the latest array sent to each canvas with
	 * the real world lights.
	 */
	public final void start()
	{
		if (!isRunning && !isRunningTest){
			isRunning = true;
			if (thread == null){
				thread = new Thread(this);
				thread.start();			
			}			
		}
	}

	/**
	 * stop autosync
	 */
	public final void stop()
	{
		isRunning = false;
	}	
	
	/**
	 * Forces a synchronization of all canvases with all recipients.  This is
	 * what run() calls- but you can call it on your own if you don't want
	 * ELU to be it's own thread.
	 */
	public void syncAllLights()
	{		
		for (Recipient r : recipients.values())
		{
			r.sync();
		}
		fpsBuffer.markFrame();
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
	public double getMeasuredFPS()
	{
		return this.fpsBuffer.getFPS();
	}
	
	/**
	 * Turn all channels on all fixtures on.
	 */
	public void allOn()
	{
		for (Recipient r : recipients.values())
		{
			r.allOn();
		}
	}
	
	/**
	 * Turn all channels on all fixtures off.
	 */
	public void allOff()
	{
		for (Recipient r : recipients.values())
		{
			r.allOff();
		}
	}

	public void on(String tag)
	{
		set(tag, (byte)255);
	}
	public void off(String tag)
	{
		set(tag, (byte)0);
	}

	public void set(String tag, byte value)
	{
		for (Recipient r : recipients.values())
		{
			for (CanvasDetector cd : r.getChannels())
			{
				if (cd != null && cd.tags.contains(tag))
				{
					cd.setValue(value);
				}
			}
		}
		this.syncAllLights();
	}

	protected void setTestVals(String tag, byte value)
	{
		for (Recipient r : recipients.values())
		{
			for (CanvasDetector cd : r.getChannels())
			{
				if (cd != null)
				{
					if(cd.tags.contains(tag))
					{
						cd.setValue(value);
					}else{
						cd.setValue((byte)0);
					}
				}
			}
		}
		this.syncAllLights();
	}

	protected void testDone()
	{
		System.out.println("testSuite complete.");
		this.isRunningTest = false;
	}
	
	public void runTest(String tiName)
	{
		if (!isRunning && !isRunningTest)
		{
			TestSuite ti = suites.get(tiName);
			if (ti == null)
			{
				logger.error("No such testSuite '" + tiName + "'");
			}else{
				isRunningTest = true;
				ti.start();
			}
		}		
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
		if (isRunning || isRunningTest)
			throw new OptionException("Cannot load while threads are running.");

		logger.info("ELUManager loading " + propFileName);
		ElectrolandProperties ep = new ElectrolandProperties(propFileName);

		recipients = new HashMap<String, Recipient>();
		canvases = new HashMap<String, ELUCanvas>();
		tests = new HashMap<String, Test>();
		suites = new HashMap<String, TestSuite>();

		HashMap<String, Fixture>fixtures = new HashMap<String, Fixture>();
		HashMap<String, FixtureType>types = new HashMap<String, FixtureType>();
		
		
		// get fps
		fps = ep.getRequiredInt("settings","global",FPS);			
		
		// parse recipients
		for (String name : ep.getObjectNames(RECIPIENT))
		{			
			try{
				Recipient r = (Recipient)ep.getRequiredClass(RECIPIENT, name, "class");
				// name, configure, store
				r.setName(name);
				r.configure(ep.getAll(RECIPIENT, name));
				recipients.put(name, r);

			}catch(ClassCastException e)
			{
				throw new OptionException(name + "' is not a Recipient.");
			}
		}
				
		// parse fixtureTypes
		for (String name : ep.getObjectNames(FIXTURE_TYPE))
		{
			types.put(name, new FixtureType(name, ep.getRequiredInt(FIXTURE_TYPE, name, "channels")));
		}
		
		// patch channels into each fixtureType (e.g., detectors)
		for (String name : ep.getObjectNames(DETECTOR))
		{
			int x = ep.getRequiredInt(DETECTOR, name, "x");
			int y = ep.getRequiredInt(DETECTOR, name, "y");
			int width = ep.getRequiredInt(DETECTOR, name, "w");
			int height = ep.getRequiredInt(DETECTOR, name, "h");

			try{
				DetectionModel dm = (DetectionModel)ep.getRequiredClass(DETECTOR, name, "model");				

				// patch information
				String ftname = ep.getRequired(DETECTOR, name, FIXTURE_TYPE);
				int index = ep.getRequiredInt(DETECTOR, name, "index");

				FixtureType ft = (FixtureType)types.get(ftname);
				if (ft== null){
					throw new OptionException("fixtureType '" + ftname + "' cannot be found for " + DETECTOR + "'" + name + "'.");
				}
				ft.detectors.set(index, new Detector(x,y,width,height,dm));

			}catch(ClassCastException e)
			{
				throw new OptionException(name + " is not a DetectionModel.");
			}

		}

		// parse canvases
		for (String name : ep.getObjectNames(CANVAS))
		{
			ELUCanvas ec = (ELUCanvas)ep.getRequiredClass(CANVAS, name, "class");
			ec.configure(ep.getAll(CANVAS, name));
			ec.setName(name);
			canvases.put(name, ec);
		}

		// parse fixtures
		for (String name : ep.getObjectNames(FIXTURE))
		{
			String typeStr = ep.getRequired(FIXTURE, name, FIXTURE_TYPE);
			FixtureType type = types.get(typeStr);
			if (type == null){
				throw new OptionException("fixtureType '" + typeStr + "' for object '" + name + "' of type 'fixture' could not be found.");
			}
			int startAddress = ep.getRequiredInt(FIXTURE, name, "startAddress");
			String recipStr = ep.getRequired(FIXTURE, name, RECIPIENT);
			Recipient recipient = recipients.get(recipStr);
			if (recipient == null){
				throw new OptionException("recipient '" + recipStr + "' for object '" + name + "' of type 'fixture' could not be found.");				
			}
			List<String> tags = ep.getOptionalArray(FIXTURE, name, "tags");
			Fixture fixture = new Fixture(name, type, startAddress, recipient, tags);
			
			fixtures.put(name, fixture);
		}
		
		// parse fixture to canvas mappings (this is the meat of everything)
		for (String name : ep.getObjectNames(CANVAS_FIXTURE))
		{
			//     * find the fixture
			String fixtureName = ep.getRequired(CANVAS_FIXTURE, name, FIXTURE);
			Fixture fixture = fixtures.get(fixtureName);
			if (fixture == null){				
				throw new OptionException("fixture '" + fixtureName + "' for object '" + name + "' of type 'canvasFixture' could not be found.");
			}

			//	   * find the canvas
			String cnvsName = ep.getRequired(CANVAS_FIXTURE, name, CANVAS);
			ELUCanvas canvas = canvases.get(cnvsName);
			if (canvas == null){				
				throw new OptionException("canvas '" + cnvsName + "' for object '" + name + "' of type 'canvasFixture' could not be found.");
			}

			// counter for recipient mappings
			int channel = fixture.startAddress;			

			//	   * find the recipient
			Recipient recipient = fixture.recipient;	

			
			//     for each prototype detector in the fixture (get from FixtureType)
			for (Detector dtr : fixture.type.detectors)
			{
				//      * create a CanvasDetector
				CanvasDetector cd = new CanvasDetector();
				cd.tags = fixture.tags;

				//      * calculate x,y based on the offset store it in the CanvasDetector
				double offsetX = ep.getRequiredDouble(CANVAS_FIXTURE, name, "x");
				double offsetY = ep.getRequiredDouble(CANVAS_FIXTURE, name, "y");

				double scaleX = ep.getRequiredDouble(CANVAS_FIXTURE, name, "xScale");
				double scaleY = ep.getRequiredDouble(CANVAS_FIXTURE, name, "yScale");				

				Rectangle boundary = new Rectangle((int)((scaleX * (dtr.x + offsetX))),
													(int)((scaleY * (dtr.y + offsetY))),
													(int)(dtr.width * scaleX),
													(int)(dtr.height * scaleY));
				cd.boundary = boundary;
				//      * store the detector model
				cd.model = dtr.model;

				// map the CanvasDetectors to pixel locations in the pixelgrab
				canvas.addDetector(cd);
				
				// map the CanvasDetector to a channel in the recipient.
				recipient.map(channel++, cd);
			}
		}

		// parse Tests
		for (String name : ep.getObjectNames(TEST))
		{
			Test test = new Test(name, ep.getRequiredArray("test", name, "tags"));
			tests.put(name, test);
		}
		
		
		// parse TestSuites
		for (String name : ep.getObjectNames(TEST_SUITE))
		{
			int fps = ep.getRequiredInt(TEST_SUITE, name, "fps");
			List<String> testStrs = ep.getRequiredArray(TEST_SUITE, name, "tests");
			ArrayList<Test> itests = new ArrayList<Test>(testStrs.size());
			
			for (String s : testStrs)
			{
				Test test = tests.get(s);
				if (test == null){
					throw new OptionException("cannot find test '" + s + "' in " + TEST_SUITE + " '" + name + "'");
				}else{
					itests.add(test);					
				}
			}

			int loops = ep.getRequiredInt(TEST_SUITE, name, "loops");
			byte color = ep.getRequiredInt(TEST_SUITE, name, "color").byteValue();
			
			TestSuite it = new TestSuite(name, this, fps, itests, loops, color);
			suites.put(name, it);
		}
		
		
		Runtime.getRuntime().addShutdownHook(new BlackOutThread(this));
		
		return this;
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
		System.out.println("FPS set to " + fps);
		
		for (ELUCanvas c : canvases.values()){
			c.debug();
		}
		
		for (Recipient r : recipients.values()){
			r.debug();
		}
		
		for (TestSuite t : suites.values())
		{
			t.debug();
		}
	}
}

class BlackOutThread extends Thread{
	private ELUManager elu;
	public BlackOutThread(ELUManager elu)
	{
		this.elu = elu;
	}
	public void run()
	{
		elu.stop();
		elu.allOff();
	}
}