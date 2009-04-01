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
import net.electroland.laface.gui.RasterPanel;
import net.electroland.laface.shows.Wave;
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
	private RasterPanel rasterPanel;
	private int guiWidth = 1060;	// TODO get from properties
	private int guiHeight = 180;

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

		rasterPanel = new RasterPanel(dmr.getRecipients());
		Raster raster = getRaster();
		rasterPanel.setRaster(raster);
		add(rasterPanel, "wrap");
		
		Animation a = new Wave(raster);
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
	
	public void actionPerformed(ActionEvent e) {
		// TODO Respond to JFrame event
		System.out.println(e.getActionCommand());
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
