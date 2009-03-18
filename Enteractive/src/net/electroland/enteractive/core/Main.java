package net.electroland.enteractive.core;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Collection;
import java.util.Properties;
import javax.swing.JFrame;
import processing.core.PApplet;
import processing.core.PConstants;
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

@SuppressWarnings("serial")
public class Main extends JFrame implements AnimationListener, ActionListener, TimedEventListener{
	
	private DetectorManager dmr;
	private DetectorManagerJPanel dmp;
	private AnimationManager amr;
	private SoundManager smr;
	private Model m;
	private PApplet p5;
	private TCUtil tcu;
	private PersonTracker ptr;
	private UDPParser udp;
	private Properties lightProps;
	
	public Main(String[] args) throws UnknownHostException, OptionException{
		try{
			lightProps = new Properties();
			lightProps.load(new FileInputStream(new File("lights.properties")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		int fps = Integer.parseInt(lightProps.getProperty("fps"));
		
		dmr = new DetectorManager(lightProps); 				// requires loading properties
		dmp = new DetectorManagerJPanel(dmr);				// panel that renders the filters
		amr = new AnimationManager(dmp, fps);				// animation manager
		amr.addListener(this);								// let me know when animations are complete
		smr = new SoundManager(null);
		this.add(dmp);
		this.setVisible(true);
		
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
		
		p5 = new PApplet();									// used solely to get PGraphics for rasters
		Animation a = new ExampleAnimation(m, getRaster(), smr);
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
		return new Raster(p5.createGraphics(0, 0, PConstants.P3D)); // or whatever it is Processing requires to generate a PGraphics.
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
