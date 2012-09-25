package net.electroland.gotham.processing.assets;

import controlP5.ControlP5;
import controlP5.ControlWindow;
import controlP5.Controller;
import controlP5.Knob;
import controlP5.Toggle;
import processing.core.PApplet;
import net.electroland.gotham.processing.EastBlurTest;

public class StripeGUIManager{

	private ControlP5 control;
	private ControlWindow window;
	private Controller<Toggle> bgColor;
	private Controller<Toggle> rSpeeds;
	private Controller<Knob> blurKnob;
	private Controller<Knob> speedKnob;

	public StripeGUIManager(PApplet p) {
		control = new ControlP5(p);

		// Init window
		window = control
				.addControlWindow("Stripe_Control_Window", 100, 100, 200, 200)
				.hideCoordinates().setBackground(p.color(90));
		// Speed Scaler Knob
		speedKnob = control.addKnob("scalerAmt").setRange(0.2f, 3.5f)
				.setValue(EastBlurTest.defaultScaler).setPosition(10, 100).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL);
		// Init blur knob
		blurKnob = control.addKnob("blurAmt").setRange(1, 100).setValue(5)
				.setPosition(100, 100).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL);
		// Init toggle switch
		bgColor = control.addToggle("blackOrWhite").setPosition(10, 40)
				.setSize(50, 20).setValue(true).setMode(ControlP5.SWITCH);

		rSpeeds = control.addToggle("randomSpeeds").setPosition(80, 40)
				.setSize(50, 20).setMode(ControlP5.SWITCH);

		// Set controls to window object
		((Toggle) bgColor).moveTo(window);
		((Knob) blurKnob).moveTo(window);
		((Knob) speedKnob).moveTo(window);
		((Toggle) rSpeeds).moveTo(window);

		//control.setBroadcast(true);
	}
}
