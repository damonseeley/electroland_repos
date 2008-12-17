package net.electroland.connection.core;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.HashMap;

import net.electroland.connection.ui.ControlWindow;

import org.apache.log4j.Logger;

/**
 * @title	"Connection" by Electroland, Indianapolis Airport, Fall 2008
 * @author	Aaron Siegel
 * @date	9-25-2008
 */

public class ConnectionMain {

	static Logger logger = Logger.getLogger(ConnectionMain.class);

	public static SoundController soundController;
	public static RenderThread renderThread;
	public static PersonTracker personTracker;
	public static ControlWindow controlWindow;
	public static HashMap<String, String> properties = new HashMap<String, String>();
	public static int linkCount = 0;
	public static float FRAMERATE = 30.0f;
	String ip;
	BufferedReader input;
	
	
	public ConnectionMain(){
		
		loadProperties();
		soundController = new SoundController(properties.get("SoundTarget").split(":")[0], Integer.parseInt(properties.get("SoundTarget").split(":")[1]));
		startWatching();
		startDrawing();
		controlWindow = new ControlWindow();
		controlWindow.setVisible(true);
	}
	
	public void loadProperties(){
		String line;
		String[] items;
		
		try{
			input = new BufferedReader(new FileReader("depends/properties.conf"));
		} catch (FileNotFoundException e){
			logger.error(e.getMessage(), e);
		}
		try{
			while((line = input.readLine()) != null){
				if(!line.startsWith("#") && line.length() > 0){	// if line is not a comment or blank...
					logger.info(line);
					items = line.split("=");					// split variable and value
					properties.put(items[0].trim(), items[1].trim());	// add to properties table
				}
			}
		} catch (IOException e){
			logger.error(e.getMessage(), e);
		}
	}
	
	public void startDrawing(){
		if(renderThread == null) { 							// don't want to start twice
			renderThread = new RenderThread(FRAMERATE);			// main draw loop
			renderThread.start();
		}
	}
	
	public void startWatching(){
		try {
			personTracker = new PersonTracker(7474);			// person tracker input
			personTracker.start();
		} catch (SocketException e) {
			logger.error(e.getMessage(), e);
		} catch (UnknownHostException e) {
			logger.error(e.getMessage(), e);
		}
	}
	
	public static void main(String[] args) {					// PROGRAM LAUNCH
		new ConnectionMain();
	}

}
