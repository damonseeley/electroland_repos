package net.electroland.gotham.processing.assets;

import net.electroland.utils.ElectrolandProperties;
import processing.core.PApplet;
import controlP5.ControlGroup;
import controlP5.ControlP5;
import controlP5.ControlWindow;
import controlP5.Controller;
import controlP5.Knob;
import controlP5.ListBox;
import controlP5.ListBoxItem;
import controlP5.Toggle;

public class StripeGUIManager{
	
	private boolean randomOnStart;
	private float defaultScaler;
	private int baseSpawnRate;
	private int baseBlurAmt;
	private float baseRandomness;

	private ControlP5 control;
	private ControlWindow window;
	private Controller<Toggle> bgColor;
	private Controller<Toggle> rSpeeds;
	private Controller<Toggle> stripesGrow;
	private Controller<Knob> blurKnob;
	private Controller<Knob> speedKnob;
	
	private Controller<Knob> howRandom;
	private Controller<Knob> howOften;
	
	private Controller<Knob> satMin;
	private Controller<Knob> satMax;
	private Controller<Knob> widthAmt;
	private Controller<Knob> colorOffset;
	
	
	private ControlGroup<ListBox> swatchList;
	
	private ElectrolandProperties props = new ElectrolandProperties("Gotham-global.properties");;

	public StripeGUIManager(PApplet p) {
		control = new ControlP5(p);
		
		//Get any initial info you need from the props file
		randomOnStart = props.getOptionalBoolean("wall", "East", "randomOnStart");
		baseSpawnRate = props.getOptionalInt("wall", "East", "baseSpawnRate");
		baseBlurAmt = props.getOptionalInt("wall", "East", "blurAmt");
		baseRandomness = (float)(props.getOptionalDouble("wall", "East", "baseRandomness")*1.0f);
		
		defaultScaler = (float) (props.getOptionalDouble("wall", "East",
				"initialScaler")*1.0f); // hacky mult by 1.0f to allow cast to float

		// Init window
		window = control
				.addControlWindow("Stripe_Control_Window", 100, 100, 400, 400)
				.hideCoordinates().setBackground(p.color(90));
		// Speed Scaler Knob
		speedKnob = control.addKnob("scalerAmt").setRange(-3.5f, 3.5f)
				.setValue(defaultScaler).setPosition(10, 200).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL)
				.setCaptionLabel("Speed");
		// Init blur knob
		blurKnob = control.addKnob("blurAmt").setRange(1, 100).setValue(baseBlurAmt)
				.setPosition(90, 200).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL)
				.setCaptionLabel("Blur Amt");
		//Randomness Offset knob
		howRandom = control.addKnob("rScaler").setRange(0, 20f)  //I think 20 might be too much. They have to start
				.setValue(baseRandomness).setPosition(170, 200).setRadius(30) //pretty far to the left (offscreen). Too high a randomness
				.setColorForeground(p.color(255))					//can make them already "behind" another stripe before it enters syncArea
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL)
				.setCaptionLabel("Amt Of Randomness");
		//SpawnRate scaler knob
		howOften = control.addKnob("spawnScaler").setRange(0, 10000f)
				.setValue(baseSpawnRate).setPosition(260, 200).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL)
				.setCaptionLabel("Spawn Rate");
		//List of Color Swatches
		swatchList = control.addListBox("whichSwatch")
			         .setPosition(250, 20)
			         .setSize(80, 150)
			         .setItemHeight(15)
			         .setColorBackground(p.color(40, 128))
			         .setColorActive(p.color(255, 128));
		swatchList.setCaptionLabel("Pick a color swatch");
		for(int i=0; i<ColorPalette.getNumSwatches(); i++){
			ListBoxItem lbi = ((ListBox) swatchList).addItem(ColorPalette.getSwatchName(i), i);//"Swatch "+i, i-1);
		    lbi.setColorBackground(p.color(0));
		}
		// Init toggle switch
		bgColor = control.addToggle("blackOrWhite").setPosition(10, 40)
				.setSize(50, 20).setValue(true).setMode(ControlP5.DEFAULT).setCaptionLabel("Black bg?");  //true = black, false = white
		stripesGrow = control.addToggle("grow").setPosition(10, 80)
				.setSize(50, 20).setValue(true).setMode(ControlP5.DEFAULT).setCaptionLabel("Pinned Stripes Grow?"); 
		rSpeeds = control.addToggle("randomSpeeds").setPosition(80, 40)
				.setSize(50, 20).setValue(randomOnStart).setMode(ControlP5.DEFAULT)
				.setCaptionLabel("Randomize Speeds?");
		
		satMin  = control.addKnob("saturationMin").setRange(0,50)
				.setValue(15).setPosition(10, 300).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL)
				.setCaptionLabel("Saturation MIN");
		satMax  = control.addKnob("saturationMax").setRange(50,100)
				.setValue(100).setPosition(90, 300).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL)
				.setCaptionLabel("Saturation MAX");		
		
		widthAmt  = control.addKnob("widthMax").setRange(1,4)
				.setValue(2).setPosition(170, 300).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL)
				.setCaptionLabel("Width MAX");	
		
		colorOffset  = control.addKnob("colOffset").setRange(0,360)
				.setValue(0).setPosition(260, 300).setRadius(30)
				.setColorForeground(p.color(255))
				.setColorBackground(p.color(200, 160, 100))
				.setColorActive(p.color(255, 60, 60))
				.setDragDirection(Knob.HORIZONTAL)
				.setCaptionLabel("Highlight Offset");	
			

		// Set controls to window object
		((Toggle) bgColor).moveTo(window);
		((Knob) blurKnob).moveTo(window);
		((Knob) speedKnob).moveTo(window);
		((Toggle) rSpeeds).moveTo(window);
		((Knob) howRandom).moveTo(window);
		((Knob) howOften).moveTo(window);
		((Toggle) stripesGrow).moveTo(window);
		swatchList.moveTo(window);
		((Knob) satMin).moveTo(window);
		((Knob) satMax).moveTo(window);
		((Knob) widthAmt).moveTo(window);
		((Knob) colorOffset).moveTo(window);

	}
}
