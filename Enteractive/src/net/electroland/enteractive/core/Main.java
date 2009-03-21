package net.electroland.enteractive.core;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Collection;
import java.util.Properties;

import javax.swing.JApplet;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import processing.core.PConstants;
import net.electroland.enteractive.gui.GUI;
import net.electroland.enteractive.gui.GUIWindow;
import net.electroland.enteractive.gui.Lights3D;
import net.electroland.enteractive.scheduler.TimedEvent;
import net.electroland.enteractive.scheduler.TimedEventListener;
import net.electroland.enteractive.shows.ExampleAnimation;
import net.electroland.lighting.detector.DetectorManagerJPanel;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.AnimationListener;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.udpUtils.TCUtil;
import net.electroland.udpUtils.UDPParser;
import net.electroland.util.OptionException;
import net.miginfocom.swing.MigLayout;

@SuppressWarnings("serial")
public class Main extends JFrame implements AnimationListener, ActionListener, TimedEventListener{
	
	private DetectorManager dmr;
	private DetectorManagerJPanel dmp;
	private AnimationManager amr;
	private SoundManager smr;
	private Model m;
	private Lights3D lights3D;
	private GUI gui;
	private TCUtil tcu;
	private PersonTracker ptr;
	private UDPParser udp;
	private Properties lightProps;
	private int guiWidth = 180;	// TODO get from properties
	private int guiHeight = 110;
	
	public Main(String[] args) throws UnknownHostException, OptionException{
		super("Enteractive Control Panel");
		try{
			lightProps = new Properties();
			lightProps.load(new FileInputStream(new File("lights.properties")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		int fps = Integer.parseInt(lightProps.getProperty("fps"));
		
		MigLayout layout = new MigLayout();
		setLayout(layout);
		//JPanel p = new JPanel(new MigLayout("", "[right]"));
		//add(p);
		
		dmr = new DetectorManager(lightProps); 				// requires loading properties
		dmp = new DetectorManagerJPanel(dmr);				// panel that renders the filters
		amr = new AnimationManager(dmp, fps);				// animation manager
		amr.addListener(this);								// let me know when animations are complete
		//smr = new SoundManager(null);
		smr = null;
		//add(dmp,"wrap");
		
		tcu = new TCUtil();									// tile controller utilities
		ptr = new PersonTracker(16,11);						// person tracker manages the model (x/y tile dimensions)
		
		try {
			udp = new UDPParser(10011, tcu, ptr);			// open our UDP receiver and begin parsing
			udp.start();
		} catch (SocketException e) {
			e.printStackTrace();
		} catch (UnknownHostException e) {
			e.printStackTrace();
		}
		
		Runtime.getRuntime().addShutdownHook(new Thread(){public void run(){tcu.billyJeanMode();}});
		
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {		// when the X is hit in the frame
				System.exit(0);								// closes app
			}
		});

		gui = new GUI(guiWidth,guiHeight);
		Raster raster = getRaster();
		((GUI)gui).setRaster(raster);
		
		JPanel lightspanel = new JPanel();
		lightspanel.setMinimumSize(new Dimension(600,600));
		lights3D = new Lights3D(600,600);
		lightspanel.add(lights3D, "west");
		add(lightspanel, "west");
		//add(gui, "wrap");
		
		JPanel placeHolder1 = new JPanel();
		placeHolder1.setSize(100, 100);
		placeHolder1.setBackground(new Color(200, 200, 200));
		placeHolder1.add(new JLabel("Controls Go Here"));
		add(placeHolder1, "wrap");
		
		JPanel placeHolder2 = new JPanel();
		placeHolder2.setSize(100, 100);
		placeHolder2.setBackground(new Color(150, 150, 150));
		placeHolder2.add(new JLabel("Raster Goes Here"));
		add(placeHolder2, "wrap");
		
		JPanel placeHolder3 = new JPanel();
		placeHolder3.setSize(100, 100);
		placeHolder3.setBackground(new Color(100, 100, 100));
		placeHolder3.add(new JLabel("Audio Levels Go Here"));
		add(placeHolder3, "wrap");
		
		setSize(800, 640);
		setVisible(true);
		setResizable(false);
		lights3D.init();
		gui.init();
		
		Animation a = new ExampleAnimation(ptr.getModel(), raster, smr);
		Collection<Recipient> fixtures = dmr.getRecipients();
		amr.startAnimation(a, fixtures); 					// start a show now, on this list of fixtures.
		amr.goLive(); 										// the whole system does nothing unless you "start" it.
	}

	public void animationComplete(Animation a) {
		// TODO Switch to a new animation
		//Animation next = ;
		//amr.startAnimation(new MyOtherAnimation(m, getRaster(), smr), f); 	// some fake animation
	}

	public void actionPerformed(ActionEvent arg0) {
		// TODO on action, force a transitioned switch
		//Animation next = new AnotherAnimation(m, getRaster(), smr); 			// some fake animation
		//amr.startAnimation(next, new FadeTransition(5), dmr.getFixtures()); 	// some fake transition with a 5 second fade
	}
	
	private Raster getRaster(){
		String[] dimensions = lightProps.getProperty("raster.tileRaster").split(" ");
		float multiplier = Float.parseFloat(lightProps.getProperty("rasterDimensionScaling"));
		int width = (int)(Integer.parseInt(dimensions[1]) * multiplier);
		int height = (int)(Integer.parseInt(dimensions[3]) * multiplier);
		return new Raster(gui.createGraphics(width, height, PConstants.P3D));
	}
	
	public Properties loadProperties(String args[]) {
		// TODO We need a lights.properties file to define our detector placement
		return null;
	}

	public void timedEvent(TimedEvent event) {
		// TODO Trigger a big event periodically
	}
	
	
	public static void main(String args[]){
		try {
			new Main(args);
		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (OptionException e) {
			e.printStackTrace();
		}
	}
}
