package net.electroland.enteractive.core;

import java.awt.Color;
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

import javax.swing.ButtonGroup;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;

import processing.core.PConstants;
import processing.core.PImage;
import net.electroland.enteractive.gui.GUI;
import net.electroland.enteractive.gui.Lights3D;
import net.electroland.enteractive.scheduler.TimedEvent;
import net.electroland.enteractive.scheduler.TimedEventListener;
import net.electroland.enteractive.shows.ExampleAnimation;
import net.electroland.enteractive.shows.LilyPad;
import net.electroland.lighting.detector.DetectorManagerJPanel;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Completable;
import net.electroland.lighting.detector.animation.CompletionListener;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.udpUtils.TCUtil;
import net.electroland.udpUtils.UDPParser;
import net.electroland.util.OptionException;
import net.miginfocom.swing.MigLayout;

@SuppressWarnings("serial")
public class Main extends JFrame implements CompletionListener, ActionListener, TimedEventListener{
	
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
	private String[] animationList;
	PImage rippleTexture, sweepTexture;
	
	public Main(String[] args) throws UnknownHostException, OptionException{
		super("Enteractive Control Panel");
		lightProps = loadProperties("lights.properties");
		int fps = Integer.parseInt(lightProps.getProperty("fps"));
		
		animationList = new String[2];
		animationList[0] = "LilyPad";
		animationList[1] = "ExampleAnimation";
		
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
		//Runtime.getRuntime().addShutdownHook(new Thread(){public void run(){tcu.billyJeanMode();}});
		
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {		// when the X is hit in the frame
				System.exit(0);								// closes app
			}
		});


		lights3D = new Lights3D(600,600, dmr.getRecipient("floor"),  dmr.getRecipient("face"));
		gui = new GUI(guiWidth,guiHeight, dmr.getRecipient("floor"),  dmr.getRecipient("face"));
		Raster raster = getRaster();
		((GUI)gui).setRaster(raster);

		drawLayout();
		lights3D.init();
		gui.init();
		
		// LOAD IMAGE SPRITES FOR SHOWS
		rippleTexture = gui.loadImage("depends//images//ripple.png");
		sweepTexture = gui.loadImage("depends//images//sweep.png");
		
		Animation a = new ExampleAnimation(ptr.getModel(), raster, smr);
		//Animation a = new LilyPad(ptr.getModel(), raster, smr, rippleTexture, sweepTexture);
		Collection<Recipient> fixtures = dmr.getRecipients();
		amr.startAnimation(a, fixtures); 					// start a show now, on this list of fixtures.
		amr.goLive(); 										// the whole system does nothing unless you "start" it.
	}

	public void completed(Completable a) {
		// TODO Switch to a new animation
		System.out.println("animation completed!");
		Raster raster = getRaster();
		((GUI)gui).setRaster(raster);
		Animation next = new LilyPad(ptr.getModel(), raster, smr, rippleTexture, sweepTexture);
		Collection<Recipient> fixtures = dmr.getRecipients();
		amr.startAnimation(next, fixtures);
		//Animation next = ;
		//amr.startAnimation(new MyOtherAnimation(m, getRaster(), smr), f); 	// some fake animation
	}

	public void actionPerformed(ActionEvent e) {
		// TODO this is temporary; needs a better method for transitioning between shows
		// TODO how to stop an animation by force???
		//System.out.println(e.getActionCommand());
		String[] event = e.getActionCommand().split(":");
		if(event[0].equals("3dmode")){
			if(event[1].equals("0")){
				lights3D.setVisible(false);
			} else if(event[1].equals("1")){
				lights3D.setVisible(true);
				lights3D.setMode(1);
			} else if(event[1].equals("2")){
				lights3D.setVisible(true);
				lights3D.setMode(2);
			}
		} else if(event[0].equals("raster")){
			if(event[1].equals("0")){
				gui.setVisible(true);
				gui.setDetectorMode(0);
			} else if(event[1].equals("1")){
				gui.setVisible(true);
				gui.setDetectorMode(1);
			} else if(event[1].equals("2")){
				gui.setVisible(true);
				gui.setDetectorMode(2);
			} else if(event[1].equals("-1")){
				gui.setVisible(false);
			}
		}
		//Animation next = new AnotherAnimation(m, getRaster(), smr); 			// some fake animation
		//amr.startAnimation(next, new FadeTransition(5), dmr.getFixtures()); 	// some fake transition with a 5 second fade
	}
	
	private void drawLayout(){
		MigLayout layout = new MigLayout("insets 0 0 0 0");
		setLayout(layout);
		//JPanel lightspanel = new JPanel();
		//lightspanel.setMinimumSize(new Dimension(600,600));
		add(lights3D, "cell 0 0 1 3");
		//lightspanel.add(lights3D, "insets 0 0 0 0");
		//lightspanel.setBackground(new Color(0, 150, 200));
		//add(lightspanel, "cell 0 0 1 3");
		
		JPanel controlPanel = new JPanel(new MigLayout());
		//controlPanel.setBackground(new Color(200, 200, 200));
		controlPanel.add(new JLabel("Current Animation:"), "wrap");
		ButtonGroup animationRadioButtons = new ButtonGroup();
		for(int i=0; i<animationList.length; i++){
			JRadioButton radio = new JRadioButton(animationList[i]);
			radio.setActionCommand("animation:"+i);
			radio.addActionListener(this);
			if(i == 0){
				radio.setSelected(true);
			}
			animationRadioButtons.add(radio);
			controlPanel.add(radio, "wrap");
		}
		
		controlPanel.add(new JLabel("3D Mode:"), "wrap");
		ButtonGroup lights3dgroup = new ButtonGroup();
		JRadioButton comparisonRadio = new JRadioButton("Comparison");
		comparisonRadio.setActionCommand("3dmode:1");
		comparisonRadio.addActionListener(this);
		comparisonRadio.setSelected(true);
		lights3dgroup.add(comparisonRadio);
		controlPanel.add(comparisonRadio, "wrap");
		JRadioButton realWorldRadio = new JRadioButton("Real World");
		realWorldRadio.setActionCommand("3dmode:2");
		realWorldRadio.addActionListener(this);
		realWorldRadio.setSelected(true);
		lights3dgroup.add(realWorldRadio);
		controlPanel.add(realWorldRadio, "wrap");
		JRadioButton disabledRadio = new JRadioButton("Disabled");
		disabledRadio.setActionCommand("3dmode:0");
		disabledRadio.addActionListener(this);
		disabledRadio.setSelected(true);
		lights3dgroup.add(disabledRadio);		
		controlPanel.add(disabledRadio, "wrap");
		
		controlPanel.add(new JLabel("Raster Mode:"), "wrap");
		ButtonGroup rastergroup = new ButtonGroup();
		JRadioButton faceRadio = new JRadioButton("Face Detectors");
		faceRadio.setActionCommand("raster:1");
		faceRadio.addActionListener(this);
		faceRadio.setSelected(true);
		rastergroup.add(faceRadio);
		controlPanel.add(faceRadio, "wrap");
		JRadioButton floorRadio = new JRadioButton("Floor Detectors");
		floorRadio.setActionCommand("raster:2");
		floorRadio.addActionListener(this);
		floorRadio.setSelected(true);
		rastergroup.add(floorRadio);
		controlPanel.add(floorRadio, "wrap");
		JRadioButton clearRadio = new JRadioButton("No Detectors");
		clearRadio.setActionCommand("raster:0");
		clearRadio.addActionListener(this);
		clearRadio.setSelected(true);
		rastergroup.add(clearRadio);
		controlPanel.add(clearRadio, "wrap");
		JRadioButton norasterRadio = new JRadioButton("Disabled");
		norasterRadio.setActionCommand("raster:-1");
		norasterRadio.addActionListener(this);
		norasterRadio.setSelected(true);
		rastergroup.add(norasterRadio);
		controlPanel.add(norasterRadio, "wrap");
		
		add(controlPanel, "cell 1 0, width 200!, height 380!, gap 0! 0! 0! 0!");
		
		JPanel rasterPanel = new JPanel();
		rasterPanel.setBackground(new Color(175, 175, 175));
		rasterPanel.add(gui);
		add(rasterPanel, "cell 1 1, width 200!, height 120!, gap 0! 0! 0! 0!");
		
		//add(gui, "cell 1 1, width 200!, height 100!, gap 0! 0! 0! 0!");
		
		JPanel placeHolder3 = new JPanel();
		placeHolder3.setBackground(new Color(150, 150, 150));
		placeHolder3.add(new JLabel("Audio Levels Go Here"));
		add(placeHolder3, "cell 1 2, width 200!, height 80!, gap 0! 0! 0! 0!");
		
		setSize(800, 620);
		setVisible(true);
		setResizable(false);
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
