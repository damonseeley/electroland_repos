package net.electroland.enteractive.core;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;
import java.util.Properties;

import javax.swing.JFrame;

import net.electroland.animation.Animation;
import net.electroland.animation.AnimationListener;
import net.electroland.animation.AnimationManager;
import net.electroland.artnet.util.DetectorManagerJPanel;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.detector.DetectorManager;

public class Main2 extends JFrame implements AnimationListener, ActionListener{

	private DetectorManager dmr;
	private DetectorManagerJPanel dmp;
	private AnimationManager amr;
	private int counter = 0;

	public static void main(String args[])
	{
		new Main2(args);
	}
	public Main2(String args[])
	{
		// ----- setup GUI -----
		// create a manager
		dmr = new DetectorManager(loadProperties(args)); // requires loading properties
		// panel that renders the filters
		dmp = new DetectorManagerJPanel(dmr);
		// animation manager
		amr = new AnimationManager(dmr);
		// let me know when animations are complete
		amr.addListener(this);

		// put the render panel into this frame
		this.add(dmp);
		// (some layout manager code required here.)
		// open the frame
		this.setVisible(true);

		// ----- Start an animation on all the current fixtures -----
		// the animation
		Animation a = new MyAnimation();

		// get all the fixtures
		List <DMXLightingFixture> fixtures = amr.getAvailableFixtures();
		amr.startAnimation(a, fixtures); // start a show now, on this list of fixtures.
		amr.goLive(); // the whole system does nothing unless you "start" it.
	}

	public Properties loadProperties(String args[])
	{
		return null;
	}
	
	public void actionPerformed(ActionEvent e){
		// on action, force a transitioned switch
		Animation next = new AnotherAnimation();
		amr.startAnimation(next, new FadeTransition(5), amr.getAvailableFixtures()); // 5 second fade
	}

	public void animationComplete(Animation a, List <DMXLightingFixture> f)
	{
		Animation next;
		switch (counter++ % 2){

			case(0):
				// start a new animation after the previous stopped.
				next = new MyOtherAnimation();

			break;
			case(1):
				// start a chain of two animations (no transition between them)
				next = new MyOtherOtherAnimation();
				next.chain(new MyChainedAnimation());

			break;
		}
		amr.startAnimation(next, f);
	}
}