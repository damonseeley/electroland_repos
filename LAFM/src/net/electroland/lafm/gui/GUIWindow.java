package net.electroland.lafm.gui;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.applet.Applet;

public class GUIWindow extends Frame{
	private static final long serialVersionUID = 1L;
	private int width = 350, height = 205;
	
	public GUIWindow(){
		super("LAFM Control Panel");						// establish name
		setSize(width, height+20);							// set frame size (+top bar)
		setLayout(new FlowLayout(FlowLayout.LEFT, 0, 0));	// left/top oriented layout
		Applet gui = new GUI(width, height);				// create processing applet
		add(gui);											// add processing gui to frame
		setResizable(false);								// static size
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {		// when the X is hit in the frame
				System.exit(0);								// closes app
			}
		});
		gui.init();
	}

}
