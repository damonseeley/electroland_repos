package net.electroland.enteractive.gui;

import java.awt.FlowLayout;
import java.awt.Frame;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import processing.core.PApplet;

@SuppressWarnings("serial")
public class GUIWindow extends Frame{
	public PApplet gui;
	
	public GUIWindow(int width, int height){
		super("Enteractive Control Panel");					// establish name
		setSize(width, height+20);							// set frame size (+top bar)
		setLayout(new FlowLayout(FlowLayout.LEFT, 0, 0));	// left/top oriented layout
		gui = new GUI(width, height);						// create processing applet
		add(gui);											// add processing gui to frame
		setResizable(false);								// static size
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {		// when the X is hit in the frame
				System.exit(0);								// closes app
			}
			public void windowIconified(WindowEvent e){
				System.out.println("iconified");
				gui.setVisible(false);
			}
			public void windowDeiconified(WindowEvent e){
				System.out.println("deiconified");
				gui.setVisible(true);
			}
		});
		gui.init();
	}

}
