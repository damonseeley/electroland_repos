package net.electroland.lighting.tools;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.Iterator;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JPanel;

import net.electroland.lighting.conductor.Conductor;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.tools.views.DetectorStates;
import net.electroland.util.OptionException;

import org.apache.log4j.Logger;

@SuppressWarnings("serial")
public class SimpleVLM extends JFrame implements ActionListener{

	private static Logger logger = Logger.getLogger(SimpleVLM.class);

	private DetectorManager dm;
	private AnimationManager am;
	private Conductor c;
	private JButton on, off, run, reload;
	private JComboBox fixtureList;
	private DetectorStates ds;
	
	// should pass in conductor to so that SystemStart is called.
	// (though, that would make it impossible NOT to use the conductor).
	// perhaps a "startable()" interface and a list of startables?

	public SimpleVLM(AnimationManager am, DetectorManager dm, Conductor c){
		this.c = c;
		this.dm = dm;
		this.am = am;
		init();
	}
	
	public SimpleVLM(AnimationManager am, DetectorManager dm)
	{
		this.dm = dm;
		this.am = am;
		init();
	}
	
	public void init()
	{
		
		// simple UI:
		//Header: [title: properties file name]
		//[drop down: recipients] [drop down: view type] (not done)
		//
		//               BIG IMAGE
		//
		//[all on] [all off] [start/stop] [reload props]
		
		
		// just render the first recipient for now.
		Recipient first = dm.getRecipients().iterator().next();
		ds = new DetectorStates(first);
		am.addRecipientRepresentation(ds);
		this.add(ds, BorderLayout.CENTER);		
		
		JPanel controls = new JPanel();
        controls.setLayout(new FlowLayout());
        on = new JButton("All on");			on.addActionListener(this);
        off = new JButton("All off");		off.addActionListener(this);
        run = new JButton("Start");		run.addActionListener(this);
        reload = new JButton("Reload");		reload.addActionListener(this);
        reload.setEnabled(false);
        this.syncRunButton();

		/* list of fixtures */
		fixtureList = new JComboBox();
		Iterator <Recipient> i = dm.getRecipients().iterator();

		while (i.hasNext())
		{
			fixtureList.addItem(i.next().getID());
		}

		fixtureList.addActionListener(this);

        controls.add(on);
        controls.add(off);
        controls.add(run);
        controls.add(reload);
		controls.add(fixtureList);
        
        this.add(controls, BorderLayout.SOUTH);

        try {
			this.setTitle(dm.getPropsFile().getCanonicalPath());
		} catch (IOException e) {
			logger.debug(e);
		}
        this.setSize(450, 500);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setVisible(true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource().equals(on))
		{
			dm.allOn();
			ds.repaint();
			
		}else if (e.getSource().equals(off))
		{
			dm.allOff();
			ds.repaint();

		}else if (e.getSource().equals(run))
		{
			if (am.isRunning()){
				// should be calling systemStop() in Conductor.
				if (c != null)
					c.stopSystem();
				else
					am.stop();
			}else{
				// should be calling systemStart() in Conductor.
				if (c != null)
					c.startSystem();
				else
					am.goLive();
				// set the latest recipient.
				ds.setRecipient(dm.getRecipient((String)fixtureList.getSelectedItem()));
			}
			syncRunButton();

		}else if (e.getSource().equals(reload))
		{
			try {
				dm.init(dm.getPropsFile().getAbsoluteFile());
				am.init(am.getFPS());
			} catch (IOException f) {
				logger.error(f);
			} catch (OptionException f) {
				logger.error(f);
			}
		}else if (e.getSource().equals(fixtureList)){
			ds.setRecipient(dm.getRecipient((String)fixtureList.getSelectedItem()));
		}
	}

	public void syncRunButton(){
		if (am.isRunning()){
			run.setText("Stop");
			on.setEnabled(false);
			off.setEnabled(false);
			reload.setEnabled(false);			
		}else{
			run.setText("Start");
			on.setEnabled(true);
			off.setEnabled(true);
			reload.setEnabled(true);
		}
	}
}