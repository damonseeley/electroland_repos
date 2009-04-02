package net.electroland.laface.core;

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
import net.electroland.laface.gui.ControlPanel;
import net.electroland.laface.gui.RasterPanel;
import net.electroland.laface.shows.DrawTest;
import net.electroland.laface.shows.TraceTest;
import net.electroland.laface.shows.WaveShow;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.DetectorManagerJPanel;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.detector.animation.Completable;
import net.electroland.lighting.detector.animation.CompletionListener;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.util.OptionException;
import net.miginfocom.swing.MigLayout;

@SuppressWarnings("serial")
public class LAFACEMain extends JFrame implements CompletionListener, ActionListener{
	
	private DetectorManager dmr;
	private DetectorManagerJPanel dmp;
	private AnimationManager amr;
	private Properties lightProps;
	public RasterPanel rasterPanel;
	private ControlPanel controlPanel;
	private int guiWidth = 1056;	// TODO get from properties
	private int guiHeight = 380;

	public LAFACEMain() throws UnknownHostException, OptionException{
		super("LAFACE Control Panel");
		setLayout(new MigLayout("insets 0 0 0 0"));
		setSize(guiWidth, guiHeight);
		
		lightProps = loadProperties("lights.properties");
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
		Raster raster = getRaster();
		rasterPanel.setRaster(raster);
		add(rasterPanel, "wrap");
		controlPanel = new ControlPanel(this);
		add(controlPanel, "wrap");
		
		Animation a = new WaveShow(raster);
		//Animation a = new TraceTest(raster, 174, 7, 10);	// light grid width + gaps
		//Animation a = new DrawTest(raster, 174, 7);			// light grid width + gaps
		Collection<Recipient> fixtures = dmr.getRecipients();
		amr.startAnimation(a, fixtures); 					// start a show now, on this list of fixtures.
		amr.goLive(); 

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
	
	private Raster getRaster(){
		String[] dimensions = lightProps.getProperty("raster.faceRaster").split(" ");
		float multiplier = Float.parseFloat(lightProps.getProperty("rasterDimensionScaling"));
		int width = (int)(Integer.parseInt(dimensions[1]) * multiplier);
		int height = (int)(Integer.parseInt(dimensions[3]) * multiplier);
		return new Raster(rasterPanel.createGraphics(width, height, PConstants.P3D));
	}
	
	public Completable getCurrentAnimation(){
		return amr.getCurrentAnimation(dmr.getRecipient("face0"));
	}
	
	public void actionPerformed(ActionEvent e) {
		// TODO Respond to JFrame event
		Completable a = amr.getCurrentAnimation(dmr.getRecipient("face0"));
		if(a instanceof WaveShow){
			String[] event = e.getActionCommand().split(":");
			/*
			if(event[0].equals("damping")){
				((WaveShow) a).setDamping(Integer.parseInt(event[1])/100.0);
			} else if(event[0].equals("nonlinearity")){
				((WaveShow) a).setNonlinearity(Integer.parseInt(event[1])/100.0);
			} else if(event[0].equals("yoffset")){
				((WaveShow) a).setYoffset(Integer.parseInt(event[1])/100.0);
			} else if(event[0].equals("dx")){
				((WaveShow) a).setDX(Integer.parseInt(event[1])/100.0);
			} else if(event[0].equals("c")){
				((WaveShow) a).setC(Integer.parseInt(event[1])/100.0);
			}
			*/
		} else if(a instanceof DrawTest){
			String[] event = e.getActionCommand().split(":");
			if(event[0].equals("turnOn")){
				((DrawTest)a).turnOn(Integer.parseInt(event[1]));
			} else if(event[0].equals("turnOff")){
				((DrawTest)a).turnOff(Integer.parseInt(event[1]));
			}
		}
	}

	public void completed(Completable a) {
		// TODO Respond to animation ending
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
