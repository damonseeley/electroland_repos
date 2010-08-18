package net.electroland.lighting.tools;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

import net.electroland.lighting.conductor.Conductor;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.tools.views.DetectorStates;

public class SimpleVLM extends JFrame implements ActionListener{

	private DetectorManager dm;
	private AnimationManager am;
	private Conductor c;
	private JButton on, off, run, reload;
	
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
		DetectorStates detectors = new DetectorStates(first);
		am.addRecipientRepresentation(detectors);
		this.add(detectors, BorderLayout.CENTER);		
//		DetectorManagerJPanel dmj = new DetectorManagerJPanel(dm);
//        am.setViewer(dmj);
//        this.add(dmj, BorderLayout.CENTER);
		
		JPanel controls = new JPanel();
        controls.setLayout(new FlowLayout());
        on = new JButton("All on");			on.addActionListener(this);
        off = new JButton("All off");		off.addActionListener(this);
        run = new JButton("Start");		run.addActionListener(this);
        reload = new JButton("Reload");		reload.addActionListener(this);
        reload.setEnabled(false);
        controls.add(on);
        controls.add(off);
        controls.add(run);
        controls.add(reload);
        this.syncRunButton();

        this.add(controls, BorderLayout.SOUTH);

        this.setSize(450, 500);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setVisible(true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource().equals(on))
		{
			dm.allOn();
		}else if (e.getSource().equals(off))
		{
			dm.allOff();
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
			}
			syncRunButton();

		}else if (e.getSource().equals(reload))
		{
			System.out.println("not implemented yet.");
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
			//reload.setEnabled(true);
		}
	}
}