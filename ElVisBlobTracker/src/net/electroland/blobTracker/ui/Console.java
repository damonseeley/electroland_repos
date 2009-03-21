package net.electroland.blobTracker.ui;

import java.awt.Color;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.PrintStream;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

@SuppressWarnings("serial")
public class Console extends JFrame implements KeyListener {

	String baseTitle;



	JTextArea textArea;

	boolean sendToSystem = true;

	boolean isPinned = false;

	public class PinnableTextArea extends JTextArea {
		public void append(String s) {
			super.append(s);

			if(! isPinned) {
				setCaretPosition(textArea.getDocument().getLength());
			}
		}
	}

	public Console(String name) {
		super(name+ "   ");
		baseTitle = name;
		textArea = new PinnableTextArea();
		textArea.setEditable(false);
		JScrollPane scrollPane = new JScrollPane(textArea);
		add(scrollPane);
		this.setSize(600, 200);


		PrintStream out = new PrintStream( new TextAreaOutStream( textArea, Color.BLACK) );
		System.setOut( out );
		
		PrintStream err = new PrintStream( new TextAreaOutStream( textArea, Color.RED) );
		System.setErr( err );


		addKeyListener(this);

		this.setVisible(true);
	}
	
	public void setBaseTitle(String s) {
		baseTitle = s;
	}

	public synchronized void addKeyListener(KeyListener l) {
		textArea.addKeyListener(l);
	}

	public void keyPressed(KeyEvent e) {
		if(e.getKeyCode() == KeyEvent.VK_P) {
			isPinned = !isPinned;
			if(isPinned) {
				setTitle(baseTitle + " . ");
			} else {
				setTitle(baseTitle + "   ");					
				textArea.setCaretPosition(textArea.getDocument().getLength());
			}

		}
	}


	public void keyReleased(KeyEvent e) {

	}


	public void keyTyped(KeyEvent e) {

	}



}
