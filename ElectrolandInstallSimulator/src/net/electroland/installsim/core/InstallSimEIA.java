package net.electroland.installsim.core;

import net.electroland.installsim.ui.EISFrame;

import org.apache.log4j.Logger;


//import javax.swing.JApplet;

public class InstallSimEIA extends Thread {
	
	static Logger logger = Logger.getLogger(InstallSimEIA.class);
	
	public static ModelEIA model;
		
	public static boolean SHOWUI;
	
	public static boolean isRunning;
	private static float framerate;

	//should this be scoped?
	private static Timer timer;

	public static long curTime = System.currentTimeMillis(); //  time of frame start to aviod lots of calls to System.getcurentTime()
	public static long elapsedTime = -1; //  time between start of cur frame and last frame to avoid re calculating passage of time allover the place

	public SoundControllerP5 sc;
	
	// constructor does all the setup work
	public InstallSimEIA() {
		
		SHOWUI = true;
		
		sc = new SoundControllerP5("127.0.0.1");
		
		model = new ModelEIA(30,0.05f,sc);
		
		//System.out.println("opening frame");
		new EISFrame(1280,300,model);

		// thread stuff
		framerate = 40;
		isRunning = true;

		timer = new Timer(framerate);
		
		start();
		logger.info("InstallSim EIA Running");

	}
	

	
	
	public void run() {
		
		timer.start();
		
		curTime = System.currentTimeMillis();
		
		while (isRunning) {
			long tmpTime = System.currentTimeMillis();
			elapsedTime = tmpTime - curTime;
			curTime = tmpTime;
		
			//update people locations and have sensors detect
			model.update();

			//paint people and sensors
			if (SHOWUI){
				EISFrame.eISPanel.repaint();
			}

			timer.block();

		}

	}


	
	
	
	//TOUCH UP THESE BASED ON INTEGRATED THREAD
	
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
			InstallSimEIA.killTheads();
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
	

	
	public static void main(String[] args) {
		new InstallSimEIA();
	}
	
	
	
}
