package net.electroland.skate.core;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Shape;
import java.awt.color.ColorSpace;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Vector;
import java.util.concurrent.CopyOnWriteArrayList;

import net.electroland.skate.ui.GUIFrame;
import net.electroland.skate.ui.GUIPanel;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.Util;
import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.InvalidPixelGrabException;
import net.electroland.utils.lighting.canvas.ELUCanvas2D;

import org.apache.log4j.Logger;


/**
 * @title	"SKATE 1.0" by Electroland, A+D Summer 2011
 * @author	Damon Seeley & Bradley Geilfuss
 */

public class SkateMain extends Thread {

	static Logger logger = Logger.getLogger(SkateMain.class);

	private ELUManager elu;
	private ELUCanvas2D canvas;

	public static boolean SHOWUI;
	public static GUIFrame guiFrame;
	public static GUIPanel guiPanel;
	int GUIWidth, GUIHeight;

	public double canvasHeight, canvasWidth; 
	public int viewHeight, viewWidth;
	public float viewScale;

	public static SoundControllerP5 soundControllerP5;

	//Thread stuff
	public static boolean isRunning;
	private static float framerate;
	private static Timer timer;
	public static long curTime = System.currentTimeMillis(); //  time of frame start to aviod lots of calls to System.getcurentTime()
	public static long elapsedTime = -1; //  time between start of cur frame and last frame to avoid re calculating passage of time allover the place

	private Map<String, SkaterSequence> sequences;
	private SkaterSequence startSequence;
		
	public SkateMain() {

		SHOWUI = true;
		GUIWidth = 540;
		GUIHeight = 600;
		guiFrame = new GUIFrame(GUIWidth,GUIHeight);
		guiPanel = new GUIPanel(GUIWidth,GUIHeight);
		//add the panel to the top of the window
		guiFrame.add(guiPanel);

		viewHeight = GUIHeight - (int)(GUIHeight*0.1);
		viewWidth = GUIWidth - (int)(GUIWidth*0.1);
		viewScale = 1.0f;


		///////// Create lighting utils
		elu = new ELUManager();
		try {
			elu.load("SkateELU.properties");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (OptionException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		canvas = (ELUCanvas2D)elu.getCanvas("my2d");
		canvasHeight = canvas.getDimensions().getHeight();
		canvasWidth = canvas.getDimensions().getWidth();

		///////// Load props and create skaters
		try {
			loadSkaterProps("SkateApp.properties");
		} catch (IOException e) {
			e.printStackTrace();
		} catch (OptionException e) {
			e.printStackTrace();
		}

		///////// Init sound controller and speakers
		soundControllerP5 = new SoundControllerP5(audioIP,10000,7770,16,audioListenerPos);



		///////// Start ELU Syncing
		elu.start();  //not used in this project because we are calling sync on demand


		// TEMP for now just add one skater
		//addRandomSkater();


		/////////////// THREAD STUFF
		isRunning = true;
		timer = new Timer(framerate);
		start();
		logger.info("Skate 1.0 started up at framerate = " + framerate);


	}

	public void run() {
		timer.start();
		curTime = System.currentTimeMillis();

		// for sequences
		SkaterSequence currSeq = startSequence;
		currSeq.startSequence(); // just marks the time it started to start the timer checks.

		while (isRunning) {

			/*
			 * Run start sequence (if present)
			 */
			if (currSeq != null)
			{
				// get any skaters that are ready to go
				List<SkaterSequenceStep> toRun = currSeq.getStartable(System.currentTimeMillis());
				// start 'em.
				for (SkaterSequenceStep step : toRun){
					try {
						// borrowed from Damon.  I don't like this!!
						Skater sk8Ref;
						sk8Ref = (Skater)((step.skater).clone());
						skaters.add(sk8Ref);
						globalSkaterCount++; // this isn't calculated by the size of the List??
						sk8Ref.startAnim();
						sk8Ref.name += globalSkaterCount; // gah?!?!?
					} catch (CloneNotSupportedException e) {
						e.printStackTrace();
					}
				}				
				// see if the current sequence needs to move on to the nextShow
				currSeq = currSeq.getCurrentSequence();
			}
			
			
			/*
			 * Animate skaters
			 */
			// Advance all skater play heads
			for (Skater sk8r : skaters) {
				sk8r.animate();
			}

			
			/*
			 * Cull dead skaters from vector
			 */
			Iterator<Skater> s = skaters.iterator();
			while (s.hasNext()){
				Skater sk8r = s.next();
				if (sk8r.animComplete) {
					skaters.remove(sk8r);
					//s.remove();
				}
			}

			/**
			 * DRAWING
			 */

			Dimension skatearea = canvas.getDimensions();

			// Create a canvas image (ci) that will be synced to ELU
			BufferedImage ci = new BufferedImage(skatearea.width,skatearea.height,ColorSpace.TYPE_RGB);
			Graphics2D gci = (Graphics2D) ci.getGraphics();
			// Draw a big black rect
			gci.setColor(new Color(0,0,0));
			gci.fillRect(0,0,ci.getWidth(),ci.getHeight());
			
			// Draw skaters, only if there are skaters
			for (Skater sk8r : skaters)
			{
				int skaterWidth = sk8r.spriteSize; // current value for sprite size, get from props instead
				
				gci.setColor(new Color(255,255,255));
				// Draw a square (for now) where the skater is located, scaled for xml file max dim
				// and then sized to the canvas width/height
				//int skaterX = (int)(sk8r.getMetricPosNow()[0]/sk8r.worldDim * skatearea.width);
				int skaterX = (int)(sk8r.getCanvas2DPosNow().x);

				//flip y to account for UCS diffs between 3DSMax and Java
				//int skaterY = (int)(sk8r.getMetricPosNow()[1]/sk8r.worldDim * skatearea.height) * -1;
				int skaterY = (int)(sk8r.getCanvas2DPosNow().y); //all xforms now contained within skater

				// SIMPLE DRAWING
				//gci.drawImage(sk8r.spriteImg, skaterX-skaterWidth/2, skaterY-skaterWidth/2, skaterWidth, skaterWidth, null);
				
				// AMPLITUDE BASED DRAWING
				// this code is hacky, need a way to draw the image once with an alpha value!!!
				
				// draw the base sprite
				gci.drawImage(sk8r.spriteImg, skaterX-skaterWidth/2, skaterY-skaterWidth/2, skaterWidth, skaterWidth, null);
				float amp = soundControllerP5.getAmpByID(sk8r.soundNodeID);
				// determine how many versions of the sprite to draw
				float drawMax = 5.0f;
				int drawIter = (int)(amp*drawMax);
				// draw the sprite drawIter more times
				for (int i=1; i < drawIter; i++) {
					gci.drawImage(sk8r.spriteImg, skaterX-skaterWidth/2, skaterY-skaterWidth/2, skaterWidth, skaterWidth, null);
				}

			}
			
			
			/*
			 * generate rgbs array to sync with ELU
			 */	
			int w = ci.getWidth(null);
			int h = ci.getHeight(null);
			int[] rgbs = new int[w*h];
			ci.getRGB(0, 0, w, h, rgbs, 0, w);

			
			try {
				CanvasDetector[] evaled = canvas.sync(rgbs);

				/*
				 * NOW draw a guide/overlay info on top
				 */
				for (Skater sk8r : skaters)
				{
					// draw text labels to show where skaters are located
					int skaterX = (int)(sk8r.getCanvas2DPosNow().x);
					//flip y to account for UCS diffs between 3DSMax and Java
					int skaterY = (int)(sk8r.getCanvas2DPosNow().y);
					gci.setColor(new Color(128,128,128));
					Font afont = new Font("afont",Font.PLAIN,10);
					gci.setFont(afont);
					gci.drawString(sk8r.name + " @f" + sk8r.curFrame, skaterX + 32, skaterY + 9);			
				}
				
				
				/*
				 * Draw the CanvasDetectors
				 */
				for (CanvasDetector d : evaled) { // draw the results of our sync.
					Shape dShape = d.getBoundary();
					// draw detector values
					int dColor = Util.unsignedByteToInt(d.getLatestState());
					gci.setColor(new Color(dColor,dColor,dColor));
					gci.fillRect(dShape.getBounds().x,dShape.getBounds().y,dShape.getBounds().width,dShape.getBounds().height);
					// draw detector outlines
					gci.setColor(new Color(0,0,128));
					gci.drawRect(dShape.getBounds().x,dShape.getBounds().y,dShape.getBounds().width,dShape.getBounds().height);
					// draw channel values
					// can I get these from ELU for detectors?
				}
				
			
			} catch (InvalidPixelGrabException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		
			//draw the final image into the jPanel GUI
			Graphics gp = guiPanel.getGraphics();
			gp.drawImage(ci, 0, 0, null);
			
			/**
			 * END DRAWING
			 */
			
			

			//Thread ops
			//logger.info(timer.sleepTime);
			timer.block();
		}

	}

	static Random generator = new Random();
	public static int globalSkaterCount = 0;

	public static void addRandomSkater() {
		Skater sk8r = skaterDefs.get(generator.nextInt(skaterDefs.size()));
		try {
			Skater sk8Ref = (Skater)sk8r.clone();
			skaters.add(sk8Ref);				
			globalSkaterCount++;
			sk8Ref.startAnim();
			sk8Ref.name += globalSkaterCount;
			//logger.info("CLONED: " + sk8Ref.name + " FROM: " + sk8r.name + " with frameLength = " + sk8r.lengthFrames);
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


	public static Vector<Skater> skaterDefs = new Vector<Skater>();
	public static CopyOnWriteArrayList<Skater> skaters = new CopyOnWriteArrayList<Skater>();
	public static boolean audioEnabled = true;
	public Point2D.Double audioListenerPos;
	public static String audioIP;
	
	public void loadSkaterProps(String skatePropsFile) throws IOException, OptionException
	{	
		Hashtable <String, Skater> skaterDict = new Hashtable <String, Skater>();
		ElectrolandProperties op = new ElectrolandProperties(skatePropsFile);
		Set<String> skaterNames = (op.getObjectNames("skater"));
		Iterator<String> iter = skaterNames.iterator();
		while (iter.hasNext()) {
			String curSkater = iter.next();
			String animFile = op.getRequired("skater",curSkater,"animFile");
			String worldDim = op.getRequired("skater",curSkater,"worldDim");
			String[] soundList = op.getOptional("skater",curSkater,"sounds").split(",");
			String sprite = op.getRequired("skater", curSkater, "sprite");
			int spriteSize = op.getRequiredInt("skater", curSkater, "spriteSize");

			
			double canvasScale = canvasWidth/Double.parseDouble(worldDim);
			
			Skater sk8r = new Skater(curSkater, animFile, worldDim, canvasScale, sprite, spriteSize, soundList);
			skaterDefs.add(sk8r);
			skaterDict.put(curSkater, sk8r);
		}
		
		// load sequences
		sequences = parseSequences(op, skaterDict);	
		
		// get global params
		framerate = op.getRequiredInt("settings", "global", "fps");
		audioEnabled = Boolean.parseBoolean(op.getRequired("settings", "global", "audio"));
		audioListenerPos = new Point2D.Double(op.getRequiredDouble("settings", "global", "listenerX"),op.getRequiredDouble("settings", "global", "listenerY"));
		audioIP = op.getRequired("settings", "global", "audioIP");
		String startStr = op.getOptional("settings", "global", "startsequence");
		if (startStr != null){
			this.startSequence = sequences.get(startStr);
		}
		
		//logger.info(audioListenerPos.x + "   " + audioListenerPos.y);
	}

	/**
	 * Parses a sequence of Skaters to be run.
	 * 
	 * @param p
	 * @param skaterDefs
	 * @return
	 * @throws OptionException
	 */
	private static Map<String, SkaterSequence> parseSequences(ElectrolandProperties p, Map<String,Skater> skaterDefs) throws OptionException
	{
		// put into dictionary first, so we can do the nextShow lookups.
		Hashtable<String, SkaterSequence> sequences = new Hashtable<String, SkaterSequence>();
		for (String name : p.getObjectNames("sequence"))
		{
			SkaterSequence newSeq = new SkaterSequence(name);
			sequences.put(name, newSeq);
		}

		// build the objects
		for (String name : p.getObjectNames("sequence"))
		{
			SkaterSequence newSeq = sequences.get(name);
			// cuelist
			List<String> cuelist = p.getRequiredArray("sequence", name, "cuelist");
			
			// make sure cuelist contains an even number of elements
			if (cuelist.size() % 2 != 0){
				throw new OptionException("cuelist for sequence '" + name + "' has an odd number of elements.");
			}
			
			for (int i = 0; i < cuelist.size(); i+=2)
			{
				String skaterId = cuelist.get(i);
				Skater skater = skaterDefs.get(skaterId);
				if (skater == null){
					throw new OptionException("cuelist for sequence '" + name + "' references unknown skater '" + skaterId + "'");
				}
				Integer delay = Integer.parseInt(cuelist.get(i + 1));
				if (delay < 0){
					throw new OptionException("cuelist for sequence '" + name + "' contains a negative delay value.");
				}
				newSeq.getSteps().add(new SkaterSequenceStep(skater, delay));
			}

			// loop (0 = no loop)
			Double loops = p.getOptionalDouble("sequence", name, "loops");
			newSeq.setDefaultLoops(loops != null ? loops.intValue() : 1);
			
			// nextShow
			String nextStr = p.getOptional("sequence", name, "nextShow");
			if (nextStr != null){
				SkaterSequence next = sequences.get(nextStr);
				if (next == null){
					throw new OptionException("sequence '" + name + "' has unknown nextShow.");					
				}
				newSeq.setNextShow(next);
			}
		}		
		return sequences;
	}

	public static void killTheads() {
		stopRunning();	
	}

	public static void stopRunning() { // it is good to have a way to stop a thread explicitly (besides System.exit(0) ) EGM
		isRunning = false;
		timer.stopRunning();
	}

	public static void restart() {
		isRunning = true;
		timer.start();
	}

	public static void shutdown() {
		try { // surround w/try catch block to make sure System.exit(0) gets call no matter what
			SkateMain.killTheads();
		} catch (Exception e) {
			e.printStackTrace();
		}
		try{ // split try/catch so if there is a problem killing threads lights will still die
			//CoopLightsMain.killLights();
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.exit(0);	
	}


	public static void main(String[] args){
		new SkateMain();
	}


}


