package net.electroland.lafm.gui;

import java.awt.FlowLayout;
import java.awt.Frame;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.Properties;

import net.electroland.lafm.core.Conductor;
import processing.core.PApplet;

public class GUIWindow extends Frame{
	private static final long serialVersionUID = 1L;
	private int width = 500, height = 320;
	public PApplet gui;
	
	public GUIWindow(Conductor conductor, Properties p){
		super("LAFM Control Panel");						// establish name
		setSize(width, height+20);							// set frame size (+top bar)
		setLayout(new FlowLayout(FlowLayout.LEFT, 0, 0));	// left/top oriented layout
		gui = new GUI(width, height, conductor, p);			// create processing applet
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
