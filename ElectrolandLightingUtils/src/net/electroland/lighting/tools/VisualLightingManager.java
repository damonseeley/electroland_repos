package net.electroland.lighting.tools;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.UnknownHostException;
import java.util.Iterator;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JScrollPane;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.util.OptionException;
import net.miginfocom.swing.MigLayout;

import org.apache.log4j.Logger;

// to do:
// save props
// load props
// open recipient window
// call "paint" on any open props for each frame in AnimationManager.run();


public class VisualLightingManager extends JFrame implements ActionListener, ChangeListener, ListSelectionListener{

	static Logger logger = Logger.getLogger(VisualLightingManager.class);

	final static String reloadStr = "Reload Props";
	final static String saveStr = "Save Props";
	final static String onStr = "All On";
	final static String offStr = "All Off";
	final static String pauseStr = "Pause";
	final static String startStr = "Start";
	final static String stopStr = "Stop";
	final static String recStr = "Recipients (double-click to show current state)";
	final static String showStr = "Shows";
	final static String desiredStr = "Desired FPS:";
	final static String measuredStr = "Measured FPS:";

	private JButton reload, save, on, off, pause, run;
	private JLabel rHead, sHead, dFPSlabel, mFPSlabel, dFPSval, mFPSval;
	private JSlider setFPS;
	private JList recipients, shows;
	private JScrollPane rPane, sPane;

	private AnimationManager am;
	private DetectorManager dm;

	// where we'll store the subframes
	private Vector<RecipientJFrame> frames = new Vector<RecipientJFrame>();

	// for testing only.
	public static void main(String args[])
	{
		String fileName = args.length == 0 ? "depends\\lights.properties" : args[0];
		try {
			new VisualLightingManager(new AnimationManager(33), new DetectorManager(fileName));
		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (OptionException e) {
			e.printStackTrace();
		}
	}

	public VisualLightingManager(AnimationManager am, DetectorManager dm)
	{
		this.am = am;
		this.dm = dm;

		// button controls
		reload = new JButton(reloadStr);		reload.addActionListener(this);
		save = new JButton(saveStr);			save.addActionListener(this);
		on = new JButton(onStr);				on.addActionListener(this);
		off = new JButton(offStr);				off.addActionListener(this);
		run = new JButton(startStr);			run.addActionListener(this);
		pause = new JButton(pauseStr);			pause.addActionListener(this);

		// Labels
		rHead = new JLabel(recStr);
		sHead = new JLabel(showStr);
		dFPSlabel = new JLabel(desiredStr, JLabel.RIGHT);
		mFPSlabel = new JLabel(measuredStr, JLabel.RIGHT);

		// fps slider
		setFPS = new JSlider(1, 100, am.getFPS()); // ISSUE: if am.getFPS() > 100, this will fail.
		setFPS.addChangeListener(this);
		
		// the labels for the FPS numerical values
		dFPSval = new JLabel("" + setFPS.getValue(), JLabel.LEFT);
		mFPSval = new JLabel("NA", JLabel.LEFT);
		
		// the recipient and show lists
		recipients = new JList();
		rPane = new JScrollPane(recipients);
		shows = new JList();
		sPane = new JScrollPane(shows);
		recipients.addListSelectionListener(this);
		
		// lay it all down.
		this.setName("test");
		this.setLayout(new MigLayout("wrap 6"));
		this.setSize(450, 500);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);		

		// (6 columns)
		// row 1
		this.add(reload);
		this.add(save);
		this.add(new JLabel(), "span 2"); // two blanks required here
		this.add(on);
		this.add(off, "wrap");

		// row 2
//		this.add(rHead, "span");

		// rows 3-9
		this.add(rPane, "span 6 6"); // (span 6 across,6 down)
		
		// row 10
		// horizontal line (span 6)

		// row 11
		this.add(run);
		this.add(pause);
		this.add(new JLabel(), "span 2"); // two blanks required here
		this.add(mFPSlabel);
		this.add(mFPSval, "wrap");
		
		// row 12
		this.add(setFPS, "span 4");
		this.add(dFPSlabel);
		this.add(dFPSval, "wrap");

		// rows 13-19
		this.add(sPane, "span 6 6"); // (span 6 across,6 down)
		
		// show it
		this.setVisible(true);
	}

	final public void start()
	{
		// disable "reloadProps", "save props", "start", "all on", "all off"
		run.setEnabled(false);
		reload.setEnabled(false);
		save.setEnabled(false);
		on.setEnabled(false);
		off.setEnabled(false);
		
		// start animation
		am.goLive();

		// enable "pause" and "stop"
		pause.setEnabled(true);
		run.setText(stopStr);
		run.setEnabled(true);
	}

	final public void stop()
	{
		// disable "stop" and "pause"
		run.setEnabled(false);
		pause.setEnabled(false);

		// stop all shows immediately
		am.stop();
		// delete existing shows
		am.init(33);

		// enable "reload props", "save props"
		reload.setEnabled(true);
		save.setEnabled(true);
		
		// crap.  should really wait on am.run() to stop here.
		// enable "start"
		run.setText(startStr);
		run.setEnabled(true);
	}

	final public void pause()
	{
		// disable "stop" and "pause"
		run.setEnabled(false);
		pause.setEnabled(false);

		// stop all shows immediately
		am.stop();

		// enable "reload props", "save props"
		reload.setEnabled(true);
		save.setEnabled(true);
		
		// crap.  should really wait on am.run() to stop here.
		// enable "start"
		run.setText(startStr);
		run.setEnabled(true);
	}

	final public void reloadProps()
	{
		// disable "start", "reload props", "save props", "pause", "all on", "all off"
		run.setEnabled(false);
		run.setText(startStr);
		reload.setEnabled(false);
		save.setEnabled(false);
		pause.setEnabled(false);
		on.setEnabled(false);
		off.setEnabled(false);

		// empty recipients window and disable it
		recipients.setEnabled(false);
		// close all recipient windows
		Iterator<RecipientJFrame>j = frames.iterator();
		while (j.hasNext())
		{
			j.next().setVisible(false);
		}
		// punt the old frames
		frames.setSize(0);

		// stop all shows immediately
		am.stop();
		// purge everything in memory
		am.init(33);

		// load the file and parse it into new objects
		try {
			dm.init(dm.getPropsFile());
			this.setTitle(dm.getPropsFile().getAbsolutePath());
		} catch (UnknownHostException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (OptionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// create the actual Frames
		Iterator<Recipient> i = dm.getRecipients().iterator();
		while (i.hasNext())
		{
			frames.add(new RecipientJFrame(i.next(), am, dm));
		}

		// populate and enable recipients window
		recipients.setListData(frames);
		recipients.setEnabled(true);

		// enable "start", "reload props", "all on", and "all off"
		run.setEnabled(true);
		reload.setEnabled(true);
		on.setEnabled(true);
		off.setEnabled(true);
	}

	final public void saveProps()
	{
		// disable "start", "reload props", "save props", "stop", "pause", "all on", "all off"
		// disable all controls on recipient panels
		// move the old file to a ~ location
//		dm.updateProps(true);
		// enable "start", "reload props", "all on", "all off"
	}

	final public void allOn()
	{
		dm.allOn();
	}

	final public void allOff()
	{
		dm.allOff();
	}

	@Override
	public void stateChanged(ChangeEvent e) 
	{
		if (e.getSource().equals(setFPS))
		{
			int fps = setFPS.getValue();
			// set label
			dFPSval.setText("" + fps);
			// set actual FPS
			am.setFPS(fps);
		}
	}

	@Override
	public void actionPerformed(ActionEvent e) 
	{
		//	private JButton reload, save, on, off, stop, pause, run;
		if (e.getSource().equals(reload))
		{
			this.reloadProps();

		} else if (e.getSource().equals(save))
		{
			this.saveProps();
			
		} else if (e.getSource().equals(on))
		{
			this.allOn();
			
		} else if (e.getSource().equals(off))
		{
			this.allOff();
		
		} else if (e.getSource().equals(run))
		{
			// this presents some potential thread problems. needs to lock
			// against am.run().
			if (am.isRunning()){
				this.stop();
			}else
			{
				this.start();
			}
			
		} else if (e.getSource().equals(pause))
		{
			this.pause();
		}
	}

	@Override
	public void valueChanged(ListSelectionEvent e) {
		if (e.getSource().equals(recipients))
		{
			Object[] toShow = recipients.getSelectedValues();
			for (int i = 0; i < toShow.length; i++)
			{
				((RecipientJFrame)toShow[i]).setVisible(true);
				((RecipientJFrame)toShow[i]).toFront();
			}
		}
	}
}