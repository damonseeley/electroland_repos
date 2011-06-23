package net.electroland.skate.core;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.Properties;

import net.electroland.skate.ui.GUIFrame;
import net.electroland.skate.ui.GUIPanel;
import net.electroland.utils.Util;
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

		Skater sx = new Skater("depends\\180f_sample.xaf");
		try {
			loadSkaterProps();
		} catch (IOException e) {
			// TODO Auto-generated catch block
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

		while (isRunning) {


			//figure out whether to add or subtract skaters

			//update sound locations

			//draw skater sprites on an image at native size
			//flop sand scale skater image to canvas-size
			//extract a pixel array from the canvas-sized sprite image and sync with ELU
			//draw detectors and skater info on the local canvas image post sync





			//test
			//BufferedImage i = new BufferedImage(400,400,ColorSpace.TYPE_RGB);
			Graphics g = guiPanel.getGraphics();
			//g.setColor(new Color(255,0,0));
			//g.fillRect((int)(20),(int)(20),20,20);
			//logger.info("Drew something?");

			BufferedImage i = new BufferedImage(400,400,ColorSpace.TYPE_RGB);
			Graphics gi = i.getGraphics();
			gi.setColor(new Color(255,150,255));
			gi.fillRect(0,0,i.getWidth(),i.getHeight());
			gi.setColor(new Color(255,0,0));
			gi.fillRect((int)(60),(int)(60),20,20);

			g.drawImage(i, 0, 0, null);









			//logger.info(timer.sleepTime);
			timer.block();
		}

	}

	public void loadSkaterProps() throws IOException
	{
		// find lights.properties
		Properties props = new Properties();
		InputStream is = new Util().getClass().getClassLoader().getResourceAsStream("depends\\Skaters.properties");
		if (is != null)
		{
			props.load(is);
		} else {
			throw new IOException("Skaters file not found!");
		}


		// parse recipients
		Enumeration <Object> g = props.keys();

		while (g.hasMoreElements())
		{
			String key = ("" + g.nextElement()).trim();
			System.out.println(key);

			/*
			if (key.toLowerCase().startsWith("recipient."))
			{
				// validate that it has an ID
				int idStart = key.indexOf('.');
				if (idStart == -1 || idStart == key.length() - 1)
				{
					throw new OptionException("no id specified in property " + key);
				}else{
					// get the ID
					String id = key.substring(idStart + 1, key.length());

					// get the props
					Map<String,String> m = OptionParser.parse("" + props.get(key));

					// load the protocol-appropriate Recipient Class.
					try {
						Recipient r = (Recipient)(new Util().getClass().getClassLoader().loadClass("" + m.get("-class")).newInstance());

						// name, configure, store
						r.setName(id);
						r.configure(m);
						recipients.put(id, r);

					// TODO: friendler error handling here.
					} catch (InstantiationException e) {
						e.printStackTrace();
					} catch (IllegalAccessException e) {
						e.printStackTrace();
					} catch (ClassNotFoundException e) {
						e.printStackTrace();
					}
				}
			}
			 */
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


