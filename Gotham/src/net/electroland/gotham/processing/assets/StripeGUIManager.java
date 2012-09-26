package net.electroland.gotham.processing.assets;

import controlP5.ControlGroup;
import controlP5.ControlP5;
import controlP5.ControlWindow;
import controlP5.Controller;
import controlP5.Knob;
import controlP5.Toggle;
import controlP5.ListBox;
import controlP5.ListBoxItem;
import processing.core.PApplet;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.utils.ElectrolandProperties;

public class StripeGUIManager{
	
	private boolean randomOnStart;
	private int defaultScaler;

	private ControlP5 control;
	private ControlWindow window;
	private Controller<Toggle> bgColor;
	private Controller<Toggle> rSpeeds;
	private Controller<Knob> blurKnob;
	private Controller<Knob> speedKnob;
	
	private Controller<Knob> howRandom;
	private Controller<Knob> howOften;
	
	private ControlGroup<ListBox> swatchList;
	
	private ElectrolandProperties props = GothamConductor.props;

	public StripeGUIManager(PApplet p) {
		control = new ControlP5(p);
		
		//Get any initial info you need from the props file
		randomOnStart = props.getOptionalBoolean("wall", "East", "randomOnStart");
		defaultScaler = props.getOptionalInt("wall", "East",
				"initialScaler");

		// Init window
		window = control
				.addControlWindow("Stripe_Control_Window", 100, 100, 400, 200)
				.hideCoordinates().setBackground(p.color(90));
		// Speed Scaler Knob
		speedKnob = control.addKnob("scalerAmt").setRange(-3.5f, 3.5f)
				.setValue(defaultScaler).setPosition(10, 100).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL)
				.setCaptionLabel("Speed");
		// Init blur knob
		blurKnob = control.addKnob("blurAmt").setRange(1, 100).setValue(5)
				.setPosition(90, 100).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL)
				.setCaptionLabel("Blur Amt");
		//Randomness Offset knob
		howRandom = control.addKnob("rScaler").setRange(0, 20f)
				.setValue(4).setPosition(170, 100).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL)
				.setCaptionLabel("Amt Of Randomness");
		//SpawnRate scaler knob
		howOften = control.addKnob("spawnScaler").setRange(0.5f, 2f)
				.setValue(1.5f).setPosition(260, 100).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL)
				.setCaptionLabel("Spawn Rate");
		//List of Color Swatches
		swatchList = control.addListBox("whichSwatch")
			         .setPosition(250, 20)
			         .setSize(80, 80)
			         .setItemHeight(15)
			         .setColorBackground(p.color(40, 128))
			         .setColorActive(p.color(255, 128));
		swatchList.setCaptionLabel("Pick a color swatch");
		for(int i=1; i<=ColorPalette.getNumSwatches(); i++){
			ListBoxItem lbi = ((ListBox) swatchList).addItem("Swatch "+i, i-1);
		    lbi.setColorBackground(p.color(0));
		}
		// Init toggle switch
		bgColor = control.addToggle("blackOrWhite").setPosition(10, 40)
				.setSize(50, 20).setValue(true).setMode(ControlP5.DEFAULT).setCaptionLabel("Black bg?");  //true = black, false = white

		rSpeeds = control.addToggle("randomSpeeds").setPosition(80, 40)
				.setSize(50, 20).setValue(randomOnStart).setMode(ControlP5.DEFAULT)
				.setCaptionLabel("Randomize Speeds?");

		// Set controls to window object
		((Toggle) bgColor).moveTo(window);
		((Knob) blurKnob).moveTo(window);
		((Knob) speedKnob).moveTo(window);
		((Toggle) rSpeeds).moveTo(window);
		((Knob) howRandom).moveTo(window);
		((Knob) howOften).moveTo(window);
		swatchList.moveTo(window);
		
		
	}
}
