package net.electroland.faces;

import java.awt.Dimension;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseListener;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Hashtable;
import java.util.Properties;
import java.util.StringTokenizer;

import javax.swing.BorderFactory;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.miginfocom.swing.MigLayout;

@SuppressWarnings("serial")
public class LAFACEApp extends JFrame {
/**
 * render panel: 2 states (for now), no resizable. 
 * 				the physics model: just show the fluid dynamics in realtime
 * 				the building face simulator: a bunch of rectangles representing
 * 					lights.  Mousing over each light should show it's channel
 * 					and universe.  In "manual" mode, clicking on rectangles
 * 					will turn them on or off.
 * 
 * label: the channel and universe of the hovered over light.  in general,
 *        a "status" console.
 * 
 * drop down menu: choose the render panel state
 * 
 * frame rate slider: 1-100 | frame rate label 1: shows selected frame rate
 * 
 * label "calculated frame rate: " | frame rate label 2: shows CALCULATED frame rate
 * 
 * threshold slider: 0-100.  The percentage of pixels for any bounding box that
 * 					must be covered by a physics model element in order to set
 * 					it's state to 'ON'
 * 
 * start/stop button: stops and starts animation
 * 
 * send to building toggle: if 'ON', the art net packets go out
 * 
 * mode radio buttons: [ ] physics [ ] test sweep [ ] manual
 * 
 * - "physics" means use the camera data to drive the physics model and the physics
 * 	model to drive the building display
 * 
 * - "test sweep" means light up one light per frame
 * 
 * - "manual" means set everything "off".  then any mouse click onto a box
 *    toggles it's state.
 * 
 * 
 * @param args
 */

	protected Image physicsModel;
	protected JPanel renderedModel;
	protected JSlider frameRateSlider, thresholdSlider;
	protected JLabel measuredFPS, requestedFPS, threshold;
	protected ButtonGroup buttonGroup;
	protected JRadioButton physicsMode, testSweepMode, manualMode;
	protected JButton startStop, allOn, allOff;
	protected JComboBox renderOptions;
	protected JCheckBox sendToBuilding;

	protected LightController[] controllers;
	private ModelThread thread;

	// should move this stuff to the controller (BuildingJPanel)
	public static final String BUILDING_MODEL = "Building Model";
	public static final String PHYSICS_MODEL = "Physics Model";
	public static final String START_STREAM = "Start Streaming";
	public static final String STOP_STREAM = "Stop Streaming";
	public static final String ALL_OFF = "All 00";
	public static final String ALL_ON = "All FF";
	public static final String DEFAULT_CONFIG_FNAME = "lights.properties";
	public static final String PHYSICS_MODE = "Physics";
	public static final String TEST_SWEEP = "Test Sweep";
	public static final String MANUAL_MODE = "Manual";
	public static final String SEND_TO_BUILDING = "Send to Building";
	
	private int default_threshold = 50;
	private int default_fps = 30;
	
	private int lastSeed = 0;
	
	public LAFACEApp(String args[]){

		super("FACES Control panel");
		init(args);
		this.setLayout(new MigLayout(""));
		
		/* the render panel for the selected model */
		renderedModel = new BuildingJPanel(true, controllers);
		renderedModel.setMinimumSize(new Dimension(1212, 100));
		// awkwardly BuildingJPanel is actually the Controller.
		renderedModel.addMouseListener((MouseListener)renderedModel);
		
		this.add(renderedModel, "span 4, growx, wrap");
		
		/* the model selector and info box */
		renderOptions = new JComboBox();
		renderOptions.addItem(BUILDING_MODEL);
		renderOptions.addItem(PHYSICS_MODEL);		
		this.add(renderOptions, "span 1");

		this.add(new JLabel(""), "gap 100, span 1");
		thresholdSlider = new JSlider(1, 99, default_threshold);
		thresholdSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				int t = thresholdSlider.getValue();
				// zero padding to keep the display from jittering.
				threshold.setText("Pixel zone treshold:" + (t < 10 ? "0" + t : t));
			}
		});
		threshold = new JLabel("Pixel zone treshold:" + thresholdSlider.getValue());
		this.add(threshold, "span 1");
		this.add(thresholdSlider, "span 1, wrap");
		
		/* runtime options */
		JPanel runtimeOptions = new JPanel();
		runtimeOptions.setBorder(BorderFactory.createTitledBorder("Runtime options"));

			/* requested frame rate (slider and value) */
			frameRateSlider = new JSlider(1, 90, default_fps);
			frameRateSlider.addChangeListener(new ChangeListener() {
				public void stateChanged(ChangeEvent e) {
					int rfps = frameRateSlider.getValue();
					// zero padding to keep the display from jittering.
					requestedFPS.setText("Requested FPS:" + (rfps < 10 ? "0" + rfps : rfps));
				}
			});
			requestedFPS = new JLabel("Requested FPS:" + frameRateSlider.getValue());
			runtimeOptions.add(requestedFPS, "span 1");
			runtimeOptions.add(frameRateSlider, "span 1");
		
			/* measured frame rate (label and value) */		
			measuredFPS = new JLabel("Measured FPS: 00");
			runtimeOptions.add(measuredFPS, "span 1");
			
			/* start / stop button & toggle for sending to the building */
			sendToBuilding = new JCheckBox(SEND_TO_BUILDING, true);
			runtimeOptions.add(sendToBuilding, "wrap");
			sendToBuilding.addActionListener((ActionListener)renderedModel);

		this.add(runtimeOptions, "span 4, center, wrap");

		allOn = new JButton(ALL_ON);
		allOff = new JButton(ALL_OFF);
		this.add(allOn, "span 2, right");		
		this.add(allOff, "span 1, center");		

		allOn.addActionListener(new ActionListener(){
	        public void actionPerformed(ActionEvent e) {

	        	// put this in a "Stop" method.  you reuse it in allOff and in manual.
				if (thread != null){	
					thread.stopThread();
				}
        		startStop.setText(START_STREAM);
	        	
	        	for (int i = 0; i < controllers.length; i++){
	        		controllers[i].allOn();
	        		controllers[i].send();
	        	}
	        	renderedModel.repaint();
	        }
		});
		
		allOff.addActionListener(new ActionListener(){
	        public void actionPerformed(ActionEvent e) {

	        	// put this in a "Stop" method.  you reuse it in allOff and in manual.
				if (thread != null){	
					thread.stopThread();
				}
        		startStop.setText(START_STREAM);

	        	for (int i = 0; i < controllers.length; i++){
	        		controllers[i].allOff();
	        		controllers[i].send();
	        	}
	        	renderedModel.repaint();	        	
	        }
		});

		
		startStop = new JButton(START_STREAM);
		this.add(startStop, "span 1, left, wrap");		

		startStop.addActionListener(new ActionListener(){
	        public void actionPerformed(ActionEvent e) {
	        	if (startStop.getText() == START_STREAM) {
	        		startStop.setText(STOP_STREAM);
	        		
	        		if (testSweepMode.isSelected()){
		        		thread = new SweepThread(lastSeed, Util.getAllLights(controllers), frameRateSlider, measuredFPS, renderedModel);
		        		thread.startThread();	        			
	        		}
	        	} else {
	        		startStop.setText(START_STREAM);
	        		if (thread != null){
	        			lastSeed = ((SweepThread)thread).getSeed();
		        		thread.stopThread();
	        		}
	        	}
	        }
		});

		/* run mode */
		JPanel buttons = new JPanel();
		buttons.setBorder(BorderFactory.createTitledBorder("Run mode"));

			buttonGroup = new ButtonGroup();

			physicsMode = new JRadioButton(PHYSICS_MODE, false);
			physicsMode.addActionListener((ActionListener)renderedModel);
			physicsMode.addActionListener(new java.awt.event.ActionListener(){
				public void actionPerformed(ActionEvent e) {
					startStop.setEnabled(true);
				}
			});
			buttons.add(physicsMode, "left");
			buttonGroup.add(physicsMode);

			testSweepMode = new JRadioButton(TEST_SWEEP, true);
			testSweepMode.addActionListener((ActionListener)renderedModel);
			testSweepMode.addActionListener(new java.awt.event.ActionListener(){
				public void actionPerformed(ActionEvent e) {
					startStop.setEnabled(true);
				}
			});

			buttons.add(testSweepMode, "center");
			buttonGroup.add(testSweepMode);

			manualMode = new JRadioButton(MANUAL_MODE, false);
			((BuildingJPanel)renderedModel).setManualMode(true);// yuck. building should poll for this.
			manualMode.addActionListener((ActionListener)renderedModel);
			manualMode.addActionListener(new java.awt.event.ActionListener(){
				public void actionPerformed(ActionEvent e) {
	        		
					if (thread != null){						
						thread.stopThread();
					}
	        		
	        		startStop.setText(START_STREAM);
					startStop.setEnabled(false);
				}
			});			
			buttons.add(manualMode, "wrap");
			buttonGroup.add(manualMode);

			
		this.add(buttons, "span 4, growx, wrap");
		

		/* activate close button for window. */
		this.addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(WindowEvent winEvt) {
		    	close();
		    }
		});

		/* show it */
		this.setSize(1250, 360);
		this.setVisible(true);
	}

	public static void main(String[] args){
		new LAFACEApp(args);
	}

	//	@SuppressWarnings("deprecation")
	private void close(){
		System.exit(0);
	}	
	
	@SuppressWarnings("unchecked")
	public void init(String[] args){
		
		// by default, read in "lights.properties"
		// unless args[] contains a different filename
		String filename = args.length == 0 ? DEFAULT_CONFIG_FNAME: args[0];

		Properties p = new Properties();
		try {
			p.load(new FileInputStream(new File(filename)));
			
			try{
				default_threshold = Integer.parseInt(p.getProperty("defaultThreshold"));
			}catch(java.lang.NumberFormatException e){
				// do nothing.  the value wasn't specified.
			}
			try{
				default_fps = Integer.parseInt(p.getProperty("defaultFPS"));
			}catch(java.lang.NumberFormatException e){
				// do nothing.  the value wasn't specified.
			}
			
			int listenerPort = p.getProperty("listenPort") == null ?
								8000 : Integer.parseInt(p.getProperty("listenPort"));
			
			// read controllers
			Hashtable a = new Hashtable();
			int controller = 0;
			while(true){
				String cdata = p.getProperty("controller" + controller++);
				if (cdata == null){
					break;
				}else{
					StringTokenizer st = new StringTokenizer(cdata, ", :");
					LightController c = new LightController((byte)Integer.parseInt(st.nextToken()),
															st.nextToken(),
															Integer.parseInt(st.nextToken()),
															listenerPort);
					a.put((int)c.universe, c);
				}
			}
			
			// read lights, and add them to controllers.
			int light = 0;
			while (true){
				String ldata = p.getProperty("light" + light++);
				if (ldata == null){
					break;
				}else{
					StringTokenizer st = new StringTokenizer(ldata, ", ");
					
					Light l = new Light(Integer.parseInt(st.nextToken()),
									Integer.parseInt(st.nextToken()),
									Integer.parseInt(st.nextToken()),
									Integer.parseInt(st.nextToken()),
									Integer.parseInt(st.nextToken()),
									Integer.parseInt(st.nextToken())
									);
					// should catch null here, and return a friendly exception.
					((LightController)a.get(l.universe)).addLight(l);
				}
			}
			
			controllers = new LightController[a.size()];
			a.values().toArray(controllers);

			// log output here.
			for (int i =0; i < controllers.length; i++){
				System.out.println(controllers[i]);
			}
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public JPanel getPhysicsModel(){
		return null;
	}

	public JPanel getBuildingModel(){
		return null;
	}
}