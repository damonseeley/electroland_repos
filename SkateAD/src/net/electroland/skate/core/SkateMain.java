package net.electroland.skate.core;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Set;
import java.util.Vector;

import net.electroland.skate.ui.GUIFrame;
import net.electroland.skate.ui.GUIPanel;
import net.electroland.utils.OptionException;
import net.electroland.utils.OptionParser;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.canvas.ELUCanvas2D;

import org.apache.log4j.Logger;


/**
 * @title	"SKATE 1.0" by Electroland, A+D Summer 2011
 * @author	Damon Seeley & Bradley Geilfuss
 */

public class SkateMain extends Thread {

	private static Logger logger = Logger.getLogger(SkateMain.class);

	private ELUManager elu;
	private ELUCanvas2D c;

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

		// create lighting utils
		//elu = new ELUManager("lights.properties");
		//c = elu.getCanvas();

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

		//init space and lights
		//init skaters
		//init sound controller and speakers

		// start everything (e.g., start the threads for each of these subsystems)
		//elu.start();

		//Skater sx = new Skater("depends//180f_sample.xaf");


		try {
			loadSkaterProps();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (OptionException e) {
			e.printStackTrace();
		}




		/////////////// THREAD STUFF
		framerate = 30;
		isRunning = true;
		timer = new Timer(framerate);
		start();
		logger.info("Skate 1.0 started up");


	}

	public void run() {
		timer.start();
		curTime = System.currentTimeMillis();

		addSkater();

		while (isRunning) {


			//figure out whether to add or subtract skaters

			//advance all skater play heads

			
			skaters.get(0).updatePlayHead();
		
			// Remove dead skaters
			if (skaters.get(0).isLive()) {
				// something
			} else {
				//skaters.remove(0);
			}


			//update sound locations
			//draw skater sprites on an image at native size
			//flop sand scale skater image to canvas-size
			//extract a pixel array from the canvas-sized sprite image and sync with ELU
			//draw detectors and skater info on the local canvas image post sync





			// JUST SHAKING OFF THE JAVA2D RUST HERE
			Graphics g = guiPanel.getGraphics();

			// Draw a big black rect on the image
			BufferedImage i = new BufferedImage(400,400,ColorSpace.TYPE_RGB);
			Graphics gi = i.getGraphics();
			gi.setColor(new Color(0,0,0));
			gi.fillRect(0,0,i.getWidth(),i.getHeight());

			if (skaters.get(0).isLive()) {
				gi.setColor(new Color(255,255,255));
				int skaterX = (int)(skaters.get(0).getMetricPosNow()[0]/skaters.get(0).maxDim * i.getWidth());
				//flip y to account for UCS diffs between 3D and Java
				int skaterY = (int)(skaters.get(0).getMetricPosNow()[1]/skaters.get(0).maxDim * i.getHeight()) * -1;
				//System.out.println(skaterX + ", " + skaterY);
				gi.fillRect(skaterX,skaterY,10,10);
				gi.setColor(new Color(128,128,128));
				gi.drawString(skaters.get(0).curFrame + "", skaterX, skaterY);			

				g.drawImage(i, 0, 0, null);
			}

			// END SHAKE





			//Thread ops
			//logger.info(timer.sleepTime);
			timer.block();
		}

	}

	public void addSkater() {
		Skater sk8r = skaterDefs.get(0);
		Skater sk8Ref;
		try {
			sk8Ref = (Skater)sk8r.clone();
			skaters.add(sk8Ref);
			sk8Ref.startAnim();
			sk8Ref.name += "--clone--";
			System.out.println("CLONED: " + sk8Ref.name + " FROM: " + sk8r.name);
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}


	public Vector<Skater> skaterDefs = new Vector<Skater>();
	public Vector<Skater> skaters = new Vector<Skater>();

	public void loadSkaterProps() throws IOException, OptionException
	{
		OptionParser op = new OptionParser("Skaters.properties");

		//System.out.println(op.getObjectNames("skater").size());

		Set<String> skaterNames = (op.getObjectNames("skater"));
		Iterator iter = skaterNames.iterator();
		while (iter.hasNext()) {
			//System.out.println(iter);
			String curSkater = iter.next().toString();
			String animFile = op.getParam("skater",curSkater,"animFile");
			String maxDim = op.getParam("skater",curSkater,"dims");
			//System.out.println(animFile);
			String[] soundList = op.getParam("skater",curSkater,"sounds").split(",");
			//System.out.println(soundList.toString());


			Skater sk8r = new Skater(curSkater, animFile, maxDim, soundList);
			skaterDefs.add(sk8r);
		}

		System.out.println(skaterDefs.get(0).name);
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


