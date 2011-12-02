package net.electroland.edmonton.core;

/**
 * By Damon Seeley
 * Should contain all UI elements for the EIA app in one place
 * except for the main image area, drawn in EIAPanel
 * 
 */

import java.awt.event.WindowEvent;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextField;

import org.apache.log4j.Logger;

@SuppressWarnings("serial")
public class EIAFrame extends JFrame  {

	protected JButton startButton, stopButton;
	protected JTextField ipAddressInput;
	protected ButtonGroup buttonGroup;
	protected JLabel ipCmds, sensorOutput;

	EIAPanel ep;

	private int windowWidth,windowHeight;
	private int panelWidth,panelHeight;

	static Logger logger = Logger.getLogger(EIAFrame.class);

	public EIAFrame(int width, int height) {
		super("EIA Setup Tool");
		
		windowWidth = width;
		windowHeight = height;

		ep = new EIAPanel(panelWidth,panelHeight);
		this.add(ep);

		//setup window
		this.setVisible(true);
		this.setSize(windowWidth, windowHeight);

		this.addWindowListener(
				new java.awt.event.WindowAdapter() {
					public void windowClosing(WindowEvent winEvt) {
						//fix
						close();
					}
				});


	}




	private void close(){
		System.exit(0);
	}



}
