package net.electroland.edmonton.core;

/**
 * By Damon Seeley
 * Should contain all UI elements for the EIA app in one place
 * except for the main image area, drawn in EIAPanel
 * 
 */

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.util.Hashtable;

import javax.swing.JButton;
import javax.swing.JFrame;

import net.electroland.ea.AnimationManager;
import net.electroland.utils.ElectrolandProperties;
import net.miginfocom.swing.MigLayout;

import org.apache.log4j.Logger;

@SuppressWarnings("serial")
public class EIAFrame extends JFrame implements ActionListener {

	EIAPanel ep;

	private int windowWidth,windowHeight;
	public Hashtable<String, Object> context;

	private int panelWidth,panelHeight;

	private JButton b1;

	static Logger logger = Logger.getLogger(EIAFrame.class);

	public EIAFrame(int width, int height, Hashtable context) {
		super("Electroland @ EIA");

		this.context = context;

		windowWidth = width;
		windowHeight = height;
		
		//setup window
		this.setVisible(true);
		this.setSize(windowWidth, windowHeight);

		ep = new EIAPanel(context);

		b1 = new JButton("Entry Shooter");
		b1.setActionCommand("entryshoot");
		b1.addActionListener(this);
		
		
		//this.setLayout(null);
		this.setLayout(new MigLayout());
		this.add(ep, "wrap");
		this.add(b1);
		

	    
		this.addWindowListener(
				new java.awt.event.WindowAdapter() {
					public void windowClosing(WindowEvent winEvt) {
						//fix
						close();
					}
				});
	}

	public void update(){
		ep.update();
	}

	public void actionPerformed(ActionEvent e) {
		if ("entryshoot".equals(e.getActionCommand())) {
			logger.info(e.getActionCommand());
			//doit
		}
	}


	private void close(){
		SoundController sc = (SoundController)context.get("soundcontroller");
		//sc.shutdown();
		System.exit(0);
	}



}
