package net.electroland.enteractive.core;

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

import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import net.electroland.enteractive.gui.GUI;
import net.electroland.enteractive.gui.Lights3D;
import net.electroland.enteractive.scheduler.TimedEvent;
import net.electroland.enteractive.scheduler.TimedEventListener;
import net.electroland.enteractive.shows.LilyPad;
import net.electroland.enteractive.shows.Pong;
import net.electroland.enteractive.shows.Spotlight;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.DetectorManagerJPanel;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.AnimationListener;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.lighting.detector.animation.transitions.LinearFade;
import net.electroland.udpUtils.TCUtil;
import net.electroland.udpUtils.UDPParser;
import net.electroland.util.OptionException;
import net.miginfocom.swing.MigLayout;
import processing.core.PConstants;
import processing.core.PImage;

@SuppressWarnings("serial")
public class EnteractiveMain extends JFrame implements AnimationListener, ActionListener, TimedEventListener, ModelListener{
	
	private DetectorManager dmr;
	private DetectorManagerJPanel dmp;
	private AnimationManager amr;
	private SoundManager smr;
	private Lights3D lights3D;
	private GUI gui;
	private TCUtil tcu;
	private PersonTracker ptr;
	private UDPParser udp;
	private Properties lightProps;
	private int guiWidth = 180;	// TODO get from properties
	private int guiHeight = 110;
	private String[] animationList;
	private JComboBox animationDropDown, displayDropDown, rasterDropDown;
	PImage rippleTexture, sweepTexture, sphereTexture, propellerTexture, spiralTexture;
	PImage ballTexture, pongTitle;	// pong textures
	
	public EnteractiveMain(String[] args) throws UnknownHostException, OptionException{
		super("Enteractive Control Panel");
		lightProps = loadProperties("lights.properties");
		int fps = Integer.parseInt(lightProps.getProperty("fps"));
		
		animationList = new String[2];
		animationList[0] = "LilyPad";
		animationList[1] = "Spotlight";
		
		dmr = new DetectorManager(lightProps); 				// requires loading properties
		dmp = new DetectorManagerJPanel(dmr);				// panel that renders the filters
		amr = new AnimationManager(dmp, fps);				// animation manager
		amr.addListener(this);								// let me know when animations are complete
		smr = new SoundManager();
		//smr = null;
		//add(dmp,"wrap");
		
		tcu = new TCUtil();									// tile controller utilities
		ptr = new PersonTracker(16,11);						// person tracker manages the model (x/y tile dimensions)
		ptr.getModel().addListener(this);					// add this as a listener
		
		try {
			udp = new UDPParser(10011, tcu, ptr);			// open our UDP receiver and begin parsing
			udp.start();
		} catch (SocketException e) {
			e.printStackTrace();
		} catch (UnknownHostException e) {
			e.printStackTrace();
		}
		Runtime.getRuntime().addShutdownHook(new Thread(){public void run(){amr.getCurrentAnimation(dmr.getRecipient("floor")).cleanUp();}});
		//Runtime.getRuntime().addShutdownHook(new Thread(){public void run(){tcu.billyJeanMode();}});
		
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {		// when the X is hit in the frame
				System.exit(0);								// closes app
			}
		});


		lights3D = new Lights3D(600,600, dmr.getRecipient("floor"),  dmr.getRecipient("face"), ptr.getModel());
		lights3D.setMinimumSize(new Dimension(600,600));
		gui = new GUI(guiWidth,guiHeight, dmr.getRecipient("floor"),  dmr.getRecipient("face"));
		Raster raster = getRaster();
		((GUI)gui).setRaster(raster);

		drawLayout();
		lights3D.init();
		gui.init();
		
		// LOAD IMAGE SPRITES FOR SHOWS
		rippleTexture = gui.loadImage("depends//images//ripple.png");
		sweepTexture = gui.loadImage("depends//images//sweep.png");
		sphereTexture = gui.loadImage("depends//images//sphere.png");
		propellerTexture = gui.loadImage("depends//images//propeller.png");
		spiralTexture = gui.loadImage("depends//images//spiral.png");
		ballTexture = gui.loadImage("depends//images//ball.png");
		pongTitle = gui.loadImage("depends//images//pongtitle.png");
		
		//currentAnimation = new ExampleAnimation(ptr.getModel(), raster, smr);
		//Animation a = new Spotlight(ptr.getModel(), raster, smr, sphereTexture);
		Animation a = new LilyPad(ptr.getModel(), raster, smr, rippleTexture, sweepTexture, propellerTexture, spiralTexture, sphereTexture);
		Collection<Recipient> fixtures = dmr.getRecipients();
		amr.startAnimation(a, fixtures); 					// start a show now, on this list of fixtures.
		amr.goLive(); 										// the whole system does nothing unless you "start" it.
	}

	public void completed(Animation a) {
		// TODO Switch to a new animation
		System.out.println("animation " + a + " completed!");
		if (a instanceof Spotlight){
			Raster raster = getRaster();
			((GUI)gui).setRaster(raster);
			Animation next = new LilyPad(ptr.getModel(), raster, smr, rippleTexture, sweepTexture, propellerTexture, spiralTexture, sphereTexture);
			Collection<Recipient> fixtures = dmr.getRecipients();
			amr.startAnimation(next, fixtures);	
		}
	}

	public void actionPerformed(ActionEvent e) {
		// TODO this is temporary; needs a better method for transitioning between shows
		// TODO how to stop an animation by force???
		//System.out.println(e.getActionCommand());
		
		if(e.getActionCommand().equals("comboBoxChanged")){
			if((JComboBox)e.getSource() == animationDropDown){
				if(animationDropDown.getSelectedItem() == "LilyPad"){
					Recipient floor = dmr.getRecipient("floor");
					if(!(amr.getCurrentAnimation(floor) instanceof LilyPad)){			// if not already lilypad
						Raster raster = getRaster();
						((GUI)gui).setRaster(raster);
						Animation a = new LilyPad(ptr.getModel(), raster, smr, rippleTexture, sweepTexture, propellerTexture, spiralTexture, sphereTexture);
						Collection<Recipient> fixtures = dmr.getRecipients();
						Animation transition = new LinearFade(2, getRaster());
						amr.startAnimation(a, transition, fixtures); 					// START LILYPAD
					}
			    } else if(animationDropDown.getSelectedItem() == "Spotlight"){			// if not already spotlight
			    	Recipient floor = dmr.getRecipient("floor");
					if(!(amr.getCurrentAnimation(floor) instanceof Spotlight)){
						Raster raster = getRaster();
						((GUI)gui).setRaster(raster);
						Animation a = new Spotlight(ptr.getModel(), raster, smr, sphereTexture);
						Collection<Recipient> fixtures = dmr.getRecipients();
						Animation transition = new LinearFade(2, getRaster());
						amr.startAnimation(a, transition, fixtures); 					// START SPOTLIGHT (30 secs)
					}
			    }
			} else if((JComboBox)e.getSource() == displayDropDown){
				if(displayDropDown.getSelectedItem() == "Comparison"){
					lights3D.setVisible(true);
					lights3D.setMode(1);
			    } else if(displayDropDown.getSelectedItem() == "Real World"){
			    	lights3D.setVisible(true);
					lights3D.setMode(2);
			    } else if(displayDropDown.getSelectedItem() == "Disabled"){
			    	lights3D.setVisible(false);
			    }
			} else if((JComboBox)e.getSource() == rasterDropDown){
				if(rasterDropDown.getSelectedItem() == "Face Detectors"){
					gui.setVisible(true);
					gui.setDetectorMode(1);
				} else if(rasterDropDown.getSelectedItem() == "Floor Detectors"){
					gui.setVisible(true);
					gui.setDetectorMode(2);
				} else if(rasterDropDown.getSelectedItem() == "No Detectors"){
					gui.setVisible(true);
					gui.setDetectorMode(0);
				} else if(rasterDropDown.getSelectedItem() == "Disabled"){
					gui.setVisible(false);
				}
			}
		    
		
		}
		//Animation next = new AnotherAnimation(m, getRaster(), smr); 			// some fake animation
		//amr.startAnimation(next, new FadeTransition(5), dmr.getFixtures()); 	// some fake transition with a 5 second fade
	}
	
	private void drawLayout(){
		MigLayout layout = new MigLayout("insets 0 0 0 0, gap 1!");
		setLayout(layout);
		setBackground(Color.black);
		setForeground(Color.white);
		add(lights3D, "cell 0 0 1 3");
		
		JPanel controlPanel = new JPanel(new MigLayout());
		controlPanel.setBackground(Color.black);
		controlPanel.setForeground(Color.white);
		controlPanel.add(new JLabel("Current Animation:"), "wrap");
		
		// drop down list to select current animation
		animationDropDown = new JComboBox(animationList);
		animationDropDown.setBackground(Color.black);
		animationDropDown.setForeground(Color.white);
		animationDropDown.addActionListener(this);		
		controlPanel.add(animationDropDown, "wrap");
		
		controlPanel.add(new JLabel("3D Mode:"), "wrap");
		
		// drop down list to select 3d mode
		displayDropDown = new JComboBox(new String[] {"Comparison", "Real World", "Disabled"});
		displayDropDown.setBackground(Color.black);
		displayDropDown.setForeground(Color.white);
		displayDropDown.addActionListener(this);		
		controlPanel.add(displayDropDown, "wrap");
		
		controlPanel.add(new JLabel("Raster Mode:"), "wrap");
		
		// drop down list to select raster mode
		rasterDropDown = new JComboBox(new String[] {"Face Detectors", "Floor Detectors", "No Detectors", "Disabled"});
		rasterDropDown.setBackground(Color.black);
		rasterDropDown.setForeground(Color.white);
		rasterDropDown.addActionListener(this);		
		controlPanel.add(rasterDropDown, "wrap");
		
		add(controlPanel, "cell 1 0, width 200!, height 380!, gap 0!");
		
		JPanel rasterPanel = new JPanel();
		rasterPanel.setBackground(Color.black);
		rasterPanel.setForeground(Color.white);
		rasterPanel.add(gui);
		add(rasterPanel, "cell 1 1, width 200!, height 120!, gap 0!");
		
		JPanel placeHolder3 = new JPanel();
		placeHolder3.setBackground(Color.black);
		placeHolder3.setForeground(Color.white);
		placeHolder3.add(new JLabel("Audio Levels Go Here"));
		add(placeHolder3, "cell 1 2, width 200!, height 90!, gap 0!");
		
		setSize(800, 620);
		setVisible(true);
		setResizable(true);
	}
	
	private Raster getRaster(){
		String[] dimensions = lightProps.getProperty("raster.tileRaster").split(" ");
		float multiplier = Float.parseFloat(lightProps.getProperty("rasterDimensionScaling"));
		int width = (int)(Integer.parseInt(dimensions[1]) * multiplier);
		int height = (int)(Integer.parseInt(dimensions[3]) * multiplier);
		return new Raster(gui.createGraphics(width, height, PConstants.P3D));
	}
	
	public Properties loadProperties(String file) {
		try{
			lightProps = new Properties();
			lightProps.load(new FileInputStream(new File(file)));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return lightProps;
	}

	public void timedEvent(TimedEvent event) {
		// TODO Trigger a big event periodically
	}

	public void modelEvent(ModelEvent e) {
		if(e.getType() == Model.ModelConstants.FOUR_CORNERS){
			System.out.println("Four Corners!");
		} else if(e.getType() == Model.ModelConstants.OPPOSITE_CORNERS){
			System.out.println("Corners 1 and 4!");
			// switch to SPOTLIGHT show
			Recipient floor = dmr.getRecipient("floor");
			if(amr.getCurrentAnimation(floor) instanceof LilyPad){
				Raster raster = getRaster();
				((GUI)gui).setRaster(raster);
				Animation a = new Spotlight(ptr.getModel(), raster, smr, sphereTexture);
				Collection<Recipient> fixtures = dmr.getRecipients();
				//Animation transition = new LinearFade(2, getRaster());
				amr.startAnimation(a, fixtures); 					// START SPOTLIGHT (30 secs)
			}
			
		} else if(e.getType() == Model.ModelConstants.OPPOSITE_CORNERS2){
			System.out.println("Corners 2 and 3!");
			// switch to PONG game
			/*
			Recipient floor = dmr.getRecipient("floor");
			if(amr.getCurrentAnimation(floor) instanceof LilyPad){
				Raster raster = getRaster();
				((GUI)gui).setRaster(raster);
				Animation a = new Pong(ptr.getModel(), raster, smr, pongTitle, ballTexture);
				Collection<Recipient> fixtures = dmr.getRecipients();
				//Animation transition = new LinearFade(2, getRaster());
				amr.startAnimation(a, fixtures); 					// START PONG (3 points)
			}
			*/
		}
	}
	
	
	public static void main(String args[]){
		try {
			new EnteractiveMain(args);
		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (OptionException e) {
			e.printStackTrace();
		}
	}
}
