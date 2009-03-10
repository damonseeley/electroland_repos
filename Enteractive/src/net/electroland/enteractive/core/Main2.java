package net.electroland.enteractive.core;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.net.UnknownHostException;
import java.util.Collection;
import java.util.List;
import java.util.Properties;

import javax.swing.JFrame;

import net.electroland.animation.Animation;
import net.electroland.animation.AnimationListener;
import net.electroland.animation.AnimationManager;
import net.electroland.animation.Raster;
import net.electroland.artnet.util.DetectorManagerJPanel;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.detector.DetectorManager;
import net.electroland.enteractive.shows.ExampleAnimation;

@SuppressWarnings("serial")
public class Main2 extends JFrame implements AnimationListener, ActionListener{

	private DetectorManager dmr;
	private DetectorManagerJPanel dmp;
	private AnimationManager amr;
	private SoundManager smr;
	private Model m;
	//private int counter = 0;

	public static void main(String args[])
	{
		try {
			new Main2(args);
		} catch (UnknownHostException e) {
			e.printStackTrace();
		}
	}
	public Main2(String args[]) throws UnknownHostException
	{
		int fps = 33;
		// ----- setup GUI -----
		// create a manager
		dmr = new DetectorManager(loadProperties(args)); // requires loading properties
		// panel that renders the filters
		dmp = new DetectorManagerJPanel(dmr);
		// animation manager
		amr = new AnimationManager(dmp, fps);
		// let me know when animations are complete
		amr.addListener(this);
		// sound manager
		// instantiate here...
		
		// put the render panel into this frame
		this.add(dmp);
		// (some layout manager code required here.)
		// open the frame
		this.setVisible(true);

		// ----- Start an animation on all the current fixtures -----
		// the animation
		Animation a = new ExampleAnimation(m, getRaster(), smr);// <-- create a raster here.

		// get all the fixtures
		Collection<DMXLightingFixture> fixtures = dmr.getFixtures();
		amr.startAnimation(a, fixtures); // start a show now, on this list of fixtures.
		amr.goLive(); // the whole system does nothing unless you "start" it.
	}

	private Raster getRaster()
	{
		// instantiate a new raster using a PImage from whereever you can get it.
		//return new Raster(this.getPGraphics()); // or whatever it is Processing requires to generate a PGraphics.
		return null;
	}

	public Properties loadProperties(String args[])
	{
		return null;
	}
	
	public void actionPerformed(ActionEvent e){
		// on action, force a transitioned switch
		//Animation next = new AnotherAnimation(m, getRaster(), smr); // some fake animation
		//amr.startAnimation(next, new FadeTransition(5), dmr.getFixtures()); // some fake transition with a 5 second fade
	}

	public void animationComplete(Animation a, List <DMXLightingFixture> f)
	{
		//Animation next = ;
		//amr.startAnimation(new MyOtherAnimation(m, getRaster(), smr), f); // some other fake animation
	}
	

	public void animationComplete(Animation a) {
		// TODO Auto-generated method stub
		
	}
}