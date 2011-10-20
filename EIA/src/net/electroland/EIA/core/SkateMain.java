package net.electroland.EIA.core;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Shape;
import java.awt.color.ColorSpace;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.awt.image.RescaleOp;
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

	public static SESSoundControllerP5 soundControllerP5;

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
		GUIWidth = 500;
		GUIHeight = 562;
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
		// channels 15 & 16 reserved for static spatial sound for now
		soundControllerP5 = new SESSoundControllerP5(audioIP,10000,7770,14,audioListenerPos);



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

	public static boolean freeze = false;
	
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
						logger.info("skaters.size():" + skaters.size());
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
				if (!freeze) {
					sk8r.animate();
				}
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
			int baseBright = (int)(baseBrightness * 255);
			gci.setColor(new Color(baseBright,baseBright,baseBright));
			gci.fillRect(0,0,ci.getWidth(),ci.getHeight());
			
			// Draw skaters, only if there are skaters
			for (Skater sk8r : skaters)
			{
				//int skaterWidth = sk8r.spriteSize; // current value for sprite size, get from props instead
				
				gci.setColor(new Color(255,255,255));
				int skaterX = (int)(sk8r.getCanvas2DPosNow().x);
				int skaterY = (int)(sk8r.getCanvas2DPosNow().y); //all xforms now contained within skater

				// SIMPLE SPRITE FOR TEST
				//gci.drawImage(sk8r.spriteImg, skaterX-skaterWidth/2, skaterY-skaterWidth/2, skaterWidth, skaterWidth, null);
				
				/**
				 * AMPLITUDE BASED DRAWING
				 * Drawing with alpha
				 * NEED to be fixed to draw in the right place
				 */
				
				// Create a rescale filter op that makes the image 50% opaque
				float amp = soundControllerP5.getAmpByID(sk8r.soundNodeID);
				//float alpha = 0.25f + amp*0.75f; //original skate calc
				//ampComponent
				//float alpha = 0.1f + amp*0.9f; //rethink calculation
				float alpha = (1.0f-ampComponent) + amp*ampComponent;
				//logger.info("Alpha = " + alpha);
				//float spriteScale = (sk8r.spriteSize*1.0f)/sk8r.spriteImg.getWidth();
				//logger.info(sk8r.spriteSize + "  " + sk8r.spriteImg.getWidth() + "  " + spriteScale);
				float[] scales = { 1.0f, 1.0f, 1.0f, alpha }; // where amp is a float from the audio system
				float[] offsets = new float[4];
				RescaleOp rop = new RescaleOp(scales, offsets, null);
				
				// Draw the image, applying the filter 
				try{
					gci.drawImage(sk8r.spriteImg, rop, skaterX-sk8r.spriteImg.getWidth()/2, skaterY-sk8r.spriteImg.getHeight()/2);					
				}catch(NullPointerException e){
					e.printStackTrace();
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
					gci.drawString("@f " + sk8r.curFrame, skaterX + 32, skaterY + 9);
					
					// draw image centers
					gci.setColor(new Color(255,0,0));
					gci.fillRect(skaterX,skaterY,2,2);
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
				}
				
				/**
				 * Draw show info
				 */
				
				gci.setColor(new Color(128,128,128));
				Font afont = new Font("afont",Font.PLAIN,10);
				gci.setFont(afont);
				double elapsedTime = currSeq.getCurrentSequence().getElapsedInSeconds();
				String nextSkater = currSeq.getCurrentSequence().getNextShow().getName();
				String currSkaters = "";
				for (Skater sk8r : skaters) {
					currSkaters += sk8r.name + ", ";
				}
				gci.drawString("sequence " + currSeq.getCurrentSequence().getName() + " @ " + elapsedTime, 10, 20);
				gci.drawString("next up: " + nextSkater, 10, 34);
				gci.drawString("current skaters: " + currSkaters, 10, 48);
				
			
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
	public static double baseBrightness;
	public static float ampComponent;
	
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
			boolean glSound = Boolean.parseBoolean(op.getRequired("skater", curSkater, "globalSound"));
			double canvasScale = canvasWidth/Double.parseDouble(worldDim);
			
			Skater sk8r = new Skater(curSkater, animFile, worldDim, canvasScale, sprite, spriteSize, soundList, glSound);

			Double duration = op.getOptionalDouble("skater", curSkater, "duration");
			if (duration != null)
			{
				sk8r.setLengthOverride(duration);
			}
			
			String reverse = op.getOptional("skater", curSkater, "reverse");
			if (reverse != null && reverse.equalsIgnoreCase("true"))
			{
				sk8r.isReversed = true;
			}
			
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
		baseBrightness = op.getRequiredDouble("settings", "global", "baseBrightness");
		ampComponent = Float.parseFloat(op.getRequired("settings", "global", "ampComponent"));
		logger.info("ampComponent = " + ampComponent);
		//System.out.println(baseBrightness);
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


