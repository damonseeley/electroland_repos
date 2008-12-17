package net.electroland.connection.ui;

import java.awt.*;
import java.awt.event.*;
import java.applet.Applet;

import net.electroland.connection.core.ConnectionMain;

public class ControlWindow extends Frame {
	private static final long serialVersionUID = 1L;
	//private int w = 1000, h = 450;
	private int w = 1010, h = 310;
	
	public ControlWindow(){
		super("Electroland Connection Installation Control Panel");		// establish name
		setSize(w, h);										// set frame size
		setLayout(new FlowLayout(FlowLayout.LEFT, 0, 0));	// left/top oriented layout
		Applet gui = new GUI(w, h);							// create processing applet
		add(gui);											// add processing gui to frame
		setResizable(false);								// static size
		
		
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {		// when the X is hit in the frame
				ConnectionMain.renderThread.lightController.sendKillPackets();	// turns lights off (don't think this runs)
				System.exit(0);								// closes app
			}
		});
		
		gui.init();											// start draw loop
	}
	
}
