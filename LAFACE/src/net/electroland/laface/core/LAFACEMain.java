package net.electroland.laface.core;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.UnknownHostException;
import java.util.Collection;
import java.util.Properties;
import javax.swing.JFrame;

import processing.core.PConstants;
import processing.core.PImage;
import net.electroland.laface.gui.ControlPanel;
import net.electroland.laface.gui.RasterPanel;
import net.electroland.laface.shows.DrawTest;
import net.electroland.laface.shows.Highlighter;
import net.electroland.laface.shows.Reflection2;
import net.electroland.laface.shows.WaveShow;
import net.electroland.laface.sprites.Wave;
import net.electroland.laface.tracking.Tracker;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.DetectorManagerJPanel;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.detector.animation.AnimationListener;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.util.OptionException;
import net.miginfocom.swing.MigLayout;

@SuppressWarnings("serial")
public class LAFACEMain extends JFrame implements AnimationListener, ActionListener{
	
	public DetectorManager dmr;
	private DetectorManagerJPanel dmp;
	public AnimationManager amr;
	private Properties lightProps;
	public RasterPanel rasterPanel;
	private ControlPanel controlPanel;
	private int guiWidth = 1056;	// TODO get from properties
	private int guiHeight = 310;
	public Raster firstRaster, secondRaster, thirdRaster;
	public CarTracker carTracker;
	public PImage highlight, linearGradient;
	public Tracker tracker;

	public LAFACEMain() throws UnknownHostException, OptionException{
		super("LAFACE Control Panel");
		setLayout(new MigLayout("insets 0 0 0 0"));
		setSize(guiWidth, guiHeight);
		
		lightProps = loadProperties("depends//lights.properties");
		int fps = Integer.parseInt(lightProps.getProperty("fps"));
		dmr = new DetectorManager(lightProps); 				// requires loading properties
		dmp = new DetectorManagerJPanel(dmr);				// panel that renders the filters
		amr = new AnimationManager(dmp, fps);				// animation manager
		amr.addListener(this);								// let me know when animations are complete
		
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {		// when the X is hit in the frame
				System.exit(0);								// closes app
			}
		});
		
		rasterPanel = new RasterPanel(this, dmr.getRecipients(), 174, 7);
		//Raster raster = getRaster();
		
		highlight = rasterPanel.loadImage("depends//images//highlight.png");
		linearGradient = rasterPanel.loadImage("depends//images//linear.png");

		firstRaster = getRaster();	// first wave show
		secondRaster = getRaster();	// second wave show
		thirdRaster = getRaster();	// transition show
		rasterPanel.setRaster(firstRaster);
		rasterPanel.setMinimumSize(new Dimension(1048,133));
		add(rasterPanel, "wrap");
		controlPanel = new ControlPanel(this);
		add(controlPanel, "wrap");

		tracker = new Tracker(this, 3);
		tracker.start();
		
		Animation a = new WaveShow(firstRaster);
		Wave wave = new Wave(0, firstRaster, 0, 0);
		((WaveShow)a).addWave(0, wave);
		Wave newwave = new Wave(1, firstRaster, 0, 0);
		newwave.setAlpha(0);	// start second wave invisible
		((WaveShow)a).addWave(1, newwave);
		// ADD ANY ADDITIONAL WAVES HERE
		
		Collection<Recipient> fixtures = dmr.getRecipients();
		amr.startAnimation(a, fixtures); 					// start the wave show
		Animation highlighter = new Highlighter(secondRaster, highlight);
		Animation newa = new Reflection2(this, thirdRaster, linearGradient);
		amr.startAnimation(highlighter, newa, fixtures);	// start a fully bright show with reflection to highlight areas
		amr.goLive(); 
		controlPanel.refreshWaveList();
		
		//Runtime.getRuntime().addShutdownHook(new Thread(){public void run(){amr.getCurrentAnimation(dmr.getRecipient("face0")).cleanUp();}});
		
		// this gets rid of exception for not using native acceleration
		System.setProperty("com.sun.media.jai.disableMediaLib", "true");
		
		// start CarTracker here
		//carTracker = new CarTracker(this);
		//carTracker.addTrackListener((TrackListener) highlighter);	// highlighter displays locations
		//carTracker.addTrackListener((TrackListener) a);
		//carTracker.start();

		setResizable(true);
		setVisible(true);
		rasterPanel.init();
	}
	
	public Properties loadProperties(String filename){
		try{
			lightProps = new Properties();
			lightProps.load(new FileInputStream(new File(filename)));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return lightProps;
	}
	
	public Raster getRaster(){
		String[] dimensions = lightProps.getProperty("raster.faceRaster").split(" ");
		float multiplier = Float.parseFloat(lightProps.getProperty("rasterDimensionScaling"));
		int width = (int)(Integer.parseInt(dimensions[1]) * multiplier);
		int height = (int)(Integer.parseInt(dimensions[3]) * multiplier);
		return new Raster(rasterPanel.createGraphics(width, height, PConstants.P3D));
	}
	
	public Animation getCurrentAnimation(){
		return amr.getCurrentAnimation(dmr.getRecipient("face0"));
	}
	
	public int getCurrentWaveID(){
		return controlPanel.getCurrentWaveID();
	}
	
	public void actionPerformed(ActionEvent e) {	// Respond to JFrame event
		Animation a = amr.getCurrentAnimation(dmr.getRecipient("face0"));
		if(a instanceof WaveShow){
			//String[] event = e.getActionCommand().split(":");
			
		} else if(a instanceof DrawTest){
			String[] event = e.getActionCommand().split(":");
			if(event[0].equals("turnOn")){
				((DrawTest)a).turnOn(Integer.parseInt(event[1]));
			} else if(event[0].equals("turnOff")){
				((DrawTest)a).turnOff(Integer.parseInt(event[1]));
			}
		}
	}

	public void completed(Animation a) {	// Respond to animation ending
		System.out.println("animation " + a + " completed!");
	}
	
	
	
	
	
	public static void main(String[] args){
		try {
			new LAFACEMain();
		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (OptionException e) {
			e.printStackTrace();
		}
	}
	
}


