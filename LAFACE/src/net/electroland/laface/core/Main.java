package net.electroland.laface.core;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.net.UnknownHostException;
import java.util.Properties;

import javax.swing.JFrame;

import processing.core.PConstants;

import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.DetectorManagerJPanel;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.detector.animation.Completable;
import net.electroland.lighting.detector.animation.CompletionListener;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.util.OptionException;
import net.miginfocom.swing.MigLayout;

@SuppressWarnings("serial")
public class Main extends JFrame implements CompletionListener, ActionListener{
	
	private DetectorManager dmr;
	private DetectorManagerJPanel dmp;
	private AnimationManager amr;
	private Properties lightProps;
	private int guiWidth = 320;	// TODO get from properties
	private int guiHeight = 240;

	public Main() throws UnknownHostException, OptionException{
		super("LAFACE Control Panel");
		setLayout(new MigLayout(""));
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

		System.out.println("Hello! I'm running!");

		setResizable(true);
		setVisible(true);
	}
	
	public Properties loadProperties(String filename){
		return null;
	}
	
//	private Raster getRaster(){
//		String[] dimensions = lightProps.getProperty("raster.tileRaster").split(" ");
//		float multiplier = Float.parseFloat(lightProps.getProperty("rasterDimensionScaling"));
//		int width = (int)(Integer.parseInt(dimensions[1]) * multiplier);
//		int height = (int)(Integer.parseInt(dimensions[3]) * multiplier);
//		return new Raster(gui.createGraphics(width, height, PConstants.P3D));
//	}
	
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
			new Main();
		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (OptionException e) {
			e.printStackTrace();
		}
	}
	
}
