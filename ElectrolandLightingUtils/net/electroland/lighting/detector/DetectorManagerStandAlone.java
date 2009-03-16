package net.electroland.lighting.detector;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.electroland.util.OptionException;
import net.miginfocom.swing.MigLayout;

@SuppressWarnings("serial")
public class DetectorManagerStandAlone extends JFrame implements ChangeListener, ActionListener {

	private DetectorManagerJPanel fixture;
	private JSlider fpsSlider;
	private JLabel fps;
	private JComboBox testAlgorithmsNames;
	private JButton sendOne, stream;
	private JCheckBox isLogging;
	private int seed = 0;
	
	private static String ON = "All ON";
	private static String OFF = "All OFF";
	private static String MODULATE = "Modulate ON/OFF";
	private static String SWEEP = "Sweep";
	private static String START_STREAM = "Start stream";
	private static String STOP_STREAM = "Stop stream";
	
	public DetectorManagerStandAlone(DetectorManager dm)
	{
		super("Fixtures");
		
		/* panel to render (and choose) a fixture */
		fixture = new DetectorManagerJPanel(dm);
		this.add(fixture);
		this.setSize(500, 500); // should make this a function of the largest fixture.
		this.setVisible(true);

		/* this frame will hold controls for test threads */
		JFrame testControls = new JFrame();
		testControls.setName("Diagnostics Tools");
		testControls.setLayout(new MigLayout(""));
		testControls.setSize(450, 150);

		/* slider for test thread fps */		
		fpsSlider = new JSlider(0, 100, 30);
		fpsSlider.addChangeListener(this);
		fps = new JLabel("30");
		testControls.add(new JLabel("FPS:"), "span 1");
		testControls.add(fpsSlider, "span 5, grow");
		testControls.add(fps, "span 1,wrap");

		/* algorithm pick list */
		testAlgorithmsNames = new JComboBox();
		testAlgorithmsNames.addItem(ON);
		testAlgorithmsNames.addItem(OFF);
		testAlgorithmsNames.addItem(MODULATE);
		testAlgorithmsNames.addItem(SWEEP);
		testAlgorithmsNames.addActionListener(this);
		testControls.add(testAlgorithmsNames, "span 3");

		/* send one packet */
		sendOne = new JButton("send one");
		sendOne.addActionListener(this);
		testControls.add(sendOne, "span 2");

		/* stream */
		stream = new JButton(START_STREAM);
		stream.addActionListener(this);
		testControls.add(stream, "span 2, wrap");

		/* toggle for logging */
		isLogging = new JCheckBox("log to console");
		isLogging.addActionListener(this);
		testControls.add(isLogging, "span 7, wrap");

		/* show the test controls */
		testControls.setVisible(true);
		
		/* activate close button for windows. */
		this.addWindowListener(new java.awt.event.WindowAdapter() 
		{
		    public void windowClosing(WindowEvent winEvt) 
		    {
				System.exit(0);
		    }
		});	
		testControls.addWindowListener(new java.awt.event.WindowAdapter() 
		{
		    public void windowClosing(WindowEvent winEvt) 
		    {
				System.exit(0);
		    }
		});	
	}

	
	protected JSlider getFPS(){
		return fpsSlider;
	}

	
	/* update vale provided by FPS slider */
	public void stateChanged(ChangeEvent e) 
	{
		if (e.getSource() == fpsSlider)
		{
			fps.setText("" + getFPS().getValue());			
		}
	}

	
	/* user picked a new thread */
	public void actionPerformed(ActionEvent e)
	{
		if (e.getSource() == testAlgorithmsNames)
		{
			this.setThread();
		}else if (e.getSource() == sendOne)
		{
			startThread(false, seed);
		}else if (e.getSource() == stream)
		{
			if (stream.getText().equals(START_STREAM)){
				stream.setText(STOP_STREAM);
				startThread(true, seed);
				sendOne.setEnabled(false);
			}else{
				stopThread();
				stream.setText(START_STREAM);				
				sendOne.setEnabled(true);
			}
		}else if (e.getSource() == isLogging){
			fixture.setLog(isLogging.isSelected());
		}
	}
	
	private void setThread(){
		// if there is a thread running, call startThread
		// (else do nothing)
	}
	
	private void startThread(boolean repeat, int seed){

		stopThread();

		// set the type of thread
		System.out.println("setting: " + testAlgorithmsNames.getSelectedItem());
		// start thread (remember to give it a link to this so it can get FPS)

		if (testAlgorithmsNames.getSelectedItem().equals(ON))
		{
			Image raster = fixture.getRaster();
			Graphics g = fixture.getRaster().getGraphics();
			g.setColor(Color.WHITE);
			g.fillRect(0, 0, raster.getWidth(fixture), raster.getHeight(fixture));
		}else if (testAlgorithmsNames.getSelectedItem().equals(OFF))
		{
			Image raster = fixture.getRaster();
			Graphics g = fixture.getRaster().getGraphics();
			g.setColor(Color.BLACK);
			g.fillRect(0, 0, raster.getWidth(fixture), raster.getHeight(fixture));
		}
		fixture.repaint();
		// if !repeat, the thread should know to immediately kill itself.

		// seed is so "send one" will step through cleanly.
	}
	
	private void stopThread(){
		
	}
	
	public static void main(String args[])
	{
		try
		{
			Properties p = new Properties();

			p.load((args.length == 1) ? 
						new FileInputStream(new File(args[0])) :
						new FileInputStream(new File("lights.properties")));			

			new DetectorManagerStandAlone(new DetectorManager(p));

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (OptionException e) {
			e.printStackTrace();
		}
	}
}