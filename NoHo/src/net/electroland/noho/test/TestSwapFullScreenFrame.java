package net.electroland.noho.test;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
 
public class TestSwapFullScreenFrame extends JFrame {
	GraphicsDevice device;
	boolean fullScreen = false;
	JButton b;
	
	public TestSwapFullScreenFrame(GraphicsDevice device) {
		super("Swap");
		setSize(400,400);
		setLocationRelativeTo(null);
		this.device = device;
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		Container c = getContentPane();
		c.setLayout(new FlowLayout(FlowLayout.CENTER,10,10));
		b = new JButton("Full screen");
		b.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				fullScreen = !fullScreen; 
				changeScreen(fullScreen);
				b.setText(fullScreen?"Windowed":"Full screen");
			}
		});		
		c.add(b);
		setVisible(true);
	}
 
	public void changeScreen(boolean full) {
		if (full) {
		// Full-screen mode
			dispose();
			setUndecorated(true);
			setResizable(false);
			device.setFullScreenWindow(this); // comment this line if you want only undecorated frame
			setVisible(true);
		} else {
		// Windowed mode
			dispose();
			setUndecorated(false);
			setResizable(true);
			device.setFullScreenWindow(null); // comment this line if you want only undecorated frame
			setVisible(true);
		}
	}
	public static void main(String[] args) {	
		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		}
		catch (Exception ex) {
			System.out.println(ex);
		}
		GraphicsEnvironment env = GraphicsEnvironment.getLocalGraphicsEnvironment();
		GraphicsDevice[] devices = env.getScreenDevices();
		TestSwapFullScreenFrame test = new TestSwapFullScreenFrame(devices[0]);
	}
}
