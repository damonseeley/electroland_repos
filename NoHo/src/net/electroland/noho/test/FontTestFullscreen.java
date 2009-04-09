package net.electroland.noho.test;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GraphicsEnvironment;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JFrame;

import net.electroland.noho.core.fontMatrix.FontMatrix;

public class FontTestFullscreen extends JFrame implements Runnable {

	private static final long serialVersionUID = 1L;

	static FontMatrix standardFont;

	int w, h; // Display height and width

	private Thread thisThread = null;

	public static void main(String[] args) {
		FontTestFullscreen ftf = new FontTestFullscreen();

	}

	public FontTestFullscreen() {

		thisThread = new Thread(this);
		thisThread.start();

		//testFont = new LetterformImage("A_2.gif");
		standardFont = new FontMatrix(
				"C:/Documents and Settings/Damon/My Documents/_ELECTROLAND/_PROJECTS/NoHo/DEV/TYPEFACE DEV/FINAL MATRIX FONT/5x7 STD GIF/");

		//		 Exiting program on window close
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				System.exit(0);
			}
		});

		// Exitig program on mouse click
		addMouseListener(new MouseListener() {
			public void mouseClicked(MouseEvent e) {
				System.exit(0);
			}

			public void mousePressed(MouseEvent e) {
			}

			public void mouseReleased(MouseEvent e) {
			}

			public void mouseEntered(MouseEvent e) {
			}

			public void mouseExited(MouseEvent e) {
				//System.exit(0);
			}
		});

		// remove window frame
		this.setUndecorated(true);

		// window should be visible
		this.setVisible(true);

		//		 switching to fullscreen mode
		GraphicsEnvironment.getLocalGraphicsEnvironment()
				.getDefaultScreenDevice().setFullScreenWindow(this);

		//      getting display resolution: width and height
		w = this.getWidth();
		h = this.getHeight();
		//System.out.println("Display resolution: " + String.valueOf(w) + "x" + String.valueOf(h));

		setSize(new Dimension(390, 14));

	}

	public void run() {
		//System.out.println("runnning");
		this.repaint();

	}

	public void paint(Graphics g) {
		//System.out.println("painting now");
		int charWidth = 10;

		String aMessage = "you can always spot the equinoxes";
		aMessage = aMessage.toLowerCase();

		for (int charNum = 0; charNum < 39; charNum++) {
			if (charNum < aMessage.length()) {
				char c = aMessage.charAt(charNum);
				//int thecharint = (int)c;
				System.out.println("printing message");

				g.drawImage(standardFont.getLetterformImg(c), charNum
						* charWidth, 0, this);
			} else {
				g.drawImage(standardFont.getLetterformImg(' '), charNum
						* charWidth, 0, this);
			}
		}

	}

	/*
	 public Dimension getPreferredSize() {
	 // just dimensions the jpanel
	 return new Dimension(390,14);
	 }
	 */

}
