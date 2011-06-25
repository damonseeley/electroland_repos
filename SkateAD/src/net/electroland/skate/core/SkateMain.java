package net.electroland.skate.core;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import net.electroland.skate.ui.GUIFrame;
import net.electroland.skate.ui.GUIPanel;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
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

			if (Math.random() < .05 ){
				addSkater();
			}


			//figure out whether to add or subtract skaters


			//advance all skater play heads
			for (Skater sk8r : skaters) {
				//System.out.println("ANIMATING: " + sk8r.name);
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





			int skateAreaWidth = (int)(GUIWidth * 0.75);
			int skateAreaHeight = (int)(GUIHeight * 0.75);


			// JUST SHAKING OFF THE JAVA2D RUST HERE
			Graphics g = guiPanel.getGraphics();

			// Draw a big black rect on the panel
			BufferedImage i = new BufferedImage(skateAreaWidth,skateAreaHeight,ColorSpace.TYPE_RGB);
			Graphics gi = i.getGraphics();
			gi.setColor(new Color(0,0,0));
			gi.fillRect(0,0,i.getWidth(),i.getHeight());

			// Draw skaters, only if there are skaters
			for (Skater sk8r : skaters)
			{
				gi.setColor(new Color(255,255,255));
				int skaterX = (int)(sk8r.getMetricPosNow()[0]/sk8r.maxDim * i.getWidth());
				//flip y to account for UCS diffs between 3D and Java
				int skaterY = (int)(sk8r.getMetricPosNow()[1]/sk8r.maxDim * i.getHeight()) * -1;
				//System.out.println(skaterX + ", " + skaterY);
				gi.fillRect(skaterX,skaterY,10,10);
				gi.setColor(new Color(128,128,128));
				gi.drawString(sk8r.curFrame + "", skaterX + 12, skaterY + 9);			

			}
			
			//draw the image no matter what to overwrite leftover frames
			g.drawImage(i, 0, 0, null);



			// END SHAKE





			//Thread ops
			//logger.info(timer.sleepTime);
			timer.block();
		}

	}

	Random generator = new Random();
	public int globalSkaterCount = 0;

	public void addSkater() {
		//System.out.println("RANDOM INT: " + generator.nextInt(skaterDefs.size()));
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

		//System.out.println(op.getObjectNames("skater").size());

		Set<String> skaterNames = (op.getObjectNames("skater"));
		Iterator iter = skaterNames.iterator();
		while (iter.hasNext()) {
			//System.out.println(iter);
			String curSkater = iter.next().toString();
			String animFile = op.getRequired("skater",curSkater,"animFile");
			String maxDim = op.getRequired("skater",curSkater,"dims");
			String[] soundList = op.getOptional("skater",curSkater,"sounds").split(",");


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


