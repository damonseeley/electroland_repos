package net.electroland.skate.core;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Shape;
import java.awt.color.ColorSpace;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import javax.swing.ImageIcon;

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


	public SkateMain() {

		SHOWUI = true;
		GUIWidth = 580;
		GUIHeight = 580;
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
		//addSkater();


		/////////////// THREAD STUFF
		isRunning = true;
		timer = new Timer(framerate);
		start();
		logger.info("Skate 1.0 started up at framerate = " + framerate);


	}

	public void run() {
		timer.start();
		curTime = System.currentTimeMillis();

		while (isRunning) {
			
			/*
			 * Determine whether to add or subtract skaters
			 */
			if (Math.random() < .02 ){
				//addSkater();
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
					s.remove();
				}
			}

			/*
			 * Rendering
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
				int skaterWidth = 64; // current value for sprite size, get from props instead
				gci.setColor(new Color(255,255,255));
				// Draw a square (for now) where the skater is located, scaled for xml file max dim
				// and then sized to the canvas width/height
				//int skaterX = (int)(sk8r.getMetricPosNow()[0]/sk8r.worldDim * skatearea.width);
				int skaterX = (int)(sk8r.getCanvas2DPosNow().x);

				//flip y to account for UCS diffs between 3DSMax and Java
				//int skaterY = (int)(sk8r.getMetricPosNow()[1]/sk8r.worldDim * skatearea.height) * -1;
				int skaterY = (int)(sk8r.getCanvas2DPosNow().y); //all xforms now contained within skater

				// Draw the square (for now)
				gci.drawImage(sk8r.spriteImg, skaterX-skaterWidth/2, skaterY-skaterWidth/2, skaterWidth, skaterWidth, null);
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





			//Thread ops
			//logger.info(timer.sleepTime);
			timer.block();
		}

	}

	static Random generator = new Random();
	public static int globalSkaterCount = 0;

	public static void addSkater() {
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
	public static Vector<Skater> skaters = new Vector<Skater>();
	public static boolean audioEnabled = true;
	public Point2D.Double audioListenerPos;
	public static String audioIP;
	
	public void loadSkaterProps(String skatePropsFile) throws IOException, OptionException
	{		
		ElectrolandProperties op = new ElectrolandProperties(skatePropsFile);
		Set<String> skaterNames = (op.getObjectNames("skater"));
		Iterator<String> iter = skaterNames.iterator();
		while (iter.hasNext()) {
			String curSkater = iter.next();
			String animFile = op.getRequired("skater",curSkater,"animFile");
			String worldDim = op.getRequired("skater",curSkater,"worldDim");
			String[] soundList = op.getOptional("skater",curSkater,"sounds").split(",");
			String sprite = op.getRequired("skater", curSkater, "sprite");
			
			double canvasScale = canvasWidth/Double.parseDouble(worldDim);
			
			Skater sk8r = new Skater(curSkater, animFile, worldDim, canvasScale, sprite, soundList);
			skaterDefs.add(sk8r);
		}
		// get global params
		framerate = op.getRequiredInt("settings", "global", "fps");
		audioEnabled = Boolean.parseBoolean(op.getRequired("settings", "global", "audio"));
		audioListenerPos = new Point2D.Double(op.getRequiredDouble("settings", "global", "listenerX"),op.getRequiredDouble("settings", "global", "listenerY"));
		audioIP = op.getRequired("settings", "global", "audioIP");
		//logger.info(audioListenerPos.x + "   " + audioListenerPos.y);
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


