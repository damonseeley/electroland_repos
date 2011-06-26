package net.electroland.skate.core;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Shape;
import java.awt.color.ColorSpace;
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

	private static Logger logger = Logger.getLogger(SkateMain.class);

	private ELUManager elu;
	private ELUCanvas2D canvas;

	public static boolean SHOWUI;
	public static GUIFrame guiFrame;
	public static GUIPanel guiPanel;
	int GUIWidth, GUIHeight;

	int canvasHeight, canvasWidth; 
	int viewHeight, viewWidth;
	float viewScale;


	//Thread stuff
	public static boolean isRunning;
	private static float framerate;
	private static Timer timer;
	public static long curTime = System.currentTimeMillis(); //  time of frame start to aviod lots of calls to System.getcurentTime()
	public static long elapsedTime = -1; //  time between start of cur frame and last frame to avoid re calculating passage of time allover the place


	public SkateMain() {

		SHOWUI = true;
		GUIWidth = 800;
		GUIHeight = 800;
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

		///////// Init sound controller and speakers


		///////// Load props and create skaters
		try {
			loadSkaterProps();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (OptionException e) {
			e.printStackTrace();
		}




		///////// Start ELU Syncing
		//elu.start();


		// TEMP for now just add one skater
		addSkater();


		/////////////// THREAD STUFF
		framerate = 45;
		isRunning = true;
		timer = new Timer(framerate);
		start();
		logger.info("Skate 1.0 started up at framerate = " + framerate);


	}

	public void run() {
		timer.start();
		curTime = System.currentTimeMillis();

		while (isRunning) {

			if (Math.random() < .01 ){
				addSkater();
			}


			/*
			 * figure out whether to add or subtract skaters
			 */

			// Advance all skater play heads
			for (Skater sk8r : skaters) {
				sk8r.animate();
			}

			// Remove dead skaters
			Iterator<Skater> s = skaters.iterator();
			while (s.hasNext()){
				Skater sk8r = s.next();
				if (sk8r.animComplete) {
					s.remove();
				}
			}

	
	
			//update sound locations
			//draw skater sprites on an image at native size
			//flop sand scale skater image to canvas-size
			//extract a pixel array from the canvas-sized sprite image and sync with ELU
			//draw detectors and skater info on the local canvas image post sync





			Dimension skatearea = canvas.getDimensions();


			/*
			 * Create a canvas image (ci) that will be synced to ELU
			 */
			BufferedImage ci = new BufferedImage(skatearea.width,skatearea.height,ColorSpace.TYPE_RGB);
			Graphics2D gci = (Graphics2D) ci.getGraphics();
			// Draw a big black rect
			gci.setColor(new Color(0,0,0));
			gci.fillRect(0,0,ci.getWidth(),ci.getHeight());

			
			Image dot = new ImageIcon("depends/whiteDot.png").getImage();
			
			// Draw skaters, only if there are skaters
			for (Skater sk8r : skaters)
			{

				gci.setColor(new Color(255,255,255));
				// Draw a square (for now) where the skater is located, scaled for xml file max dim
				// and then sized to the canvas width/height
				int skaterX = (int)(sk8r.getMetricPosNow()[0]/sk8r.maxDim * skatearea.width);
				//flip y to account for UCS diffs between 3DSMax and Java
				int skaterY = (int)(sk8r.getMetricPosNow()[1]/sk8r.maxDim * skatearea.height) * -1;
				// Draw the square (for now)
				//gci.fillOval(skaterX,skaterY,10,10);	
				gci.drawImage(dot, skaterX, skaterY, 48, 48, null);
			}
			
			
			
			/*
			 * Sync ci image with ELU
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
					int skaterX = (int)(sk8r.getMetricPosNow()[0]/sk8r.maxDim * skatearea.width);
					//flip y to account for UCS diffs between 3DSMax and Java
					int skaterY = (int)(sk8r.getMetricPosNow()[1]/sk8r.maxDim * skatearea.height) * -1;
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
					gci.setColor(new Color(d.getLatestState(),d.getLatestState(),d.getLatestState()));
					gci.fillRect(dShape.getBounds().x,dShape.getBounds().y,dShape.getBounds().width,dShape.getBounds().height);
					gci.setColor(new Color(0,0,128));
					gci.drawRect(dShape.getBounds().x,dShape.getBounds().y,dShape.getBounds().width,dShape.getBounds().height);
					
				}
				//System.out.println(canvas.getDetectors().length);				
			
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

	Random generator = new Random();
	public int globalSkaterCount = 0;

	public void addSkater() {
		Skater sk8r = skaterDefs.get(generator.nextInt(skaterDefs.size()));
		try {
			Skater sk8Ref = (Skater)sk8r.clone();
			skaters.add(sk8Ref);
			globalSkaterCount++;
			sk8Ref.startAnim();
			sk8Ref.name += globalSkaterCount;
			logger.info("CLONED: " + sk8Ref.name + " FROM: " + sk8r.name + " with frameLength = " + sk8r.lengthFrames);
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


	public Vector<Skater> skaterDefs = new Vector<Skater>();
//	public Set<Skater> skaters = new CopyOnWriteArraySet<Skater>();
	public Vector<Skater> skaters = new Vector<Skater>();
	
	public void loadSkaterProps() throws IOException, OptionException
	{
		ElectrolandProperties op = new ElectrolandProperties("Skaters.properties");
		Set<String> skaterNames = (op.getObjectNames("skater"));
		Iterator iter = skaterNames.iterator();
		while (iter.hasNext()) {
			String curSkater = iter.next().toString();
			String animFile = op.getRequired("skater",curSkater,"animFile");
			String maxDim = op.getRequired("skater",curSkater,"dims");
			String[] soundList = op.getOptional("skater",curSkater,"sounds").split(",");


			Skater sk8r = new Skater(curSkater, animFile, maxDim, soundList);
			skaterDefs.add(sk8r);
		}
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


