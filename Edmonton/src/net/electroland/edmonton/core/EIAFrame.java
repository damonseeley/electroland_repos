package net.electroland.edmonton.core;

/**
 * By Damon Seeley
 * Should contain all UI elements for the EIA app in one place
 * except for the main image area, drawn in EIAPanel
 * 
 */

import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

import net.miginfocom.swing.MigLayout;

import org.apache.log4j.Logger;

@SuppressWarnings("serial")
public class EIAFrame extends JFrame implements ActionListener {

	EIAPanel ep;

	private int windowWidth,windowHeight;
	public Hashtable<String, Object> context;

	private int panelWidth,panelHeight;

	private JButton b1,b2,b3,b4,b5,b6,b7,b8,b9,b10;
	private ArrayList<JButton> buttons;

	static Logger logger = Logger.getLogger(EIAFrame.class);

	public EIAFrame(int width, int height, Hashtable context) {
		super("Electroland @ EIA");

		this.context = context;

		windowWidth = width;
		windowHeight = height;
		
		ep = new EIAPanel(context);
		
		//Font newButtonFont = new Font("Default",Font.PLAIN,13);  
		
		buttons = new ArrayList<JButton>();
		
		b2 = new JButton("Big Fill");
		b2.setActionCommand("bigfill");
		buttons.add(b2);
		//b2.addActionListener(this);
		
		b1 = new JButton("Entry1");
		b1.setActionCommand("entry1");
		buttons.add(b1);
		//b1.addActionListener(this);
		
		b3 = new JButton("Egg1");
		b3.setActionCommand("egg1");
		buttons.add(b3);
		//b3.addActionListener(this);
		
		b6 = new JButton("Egg2");
		b6.setActionCommand("egg2");
		buttons.add(b6);
		//b3.addActionListener(this);
		
		b4 = new JButton("Exit1");
		b4.setActionCommand("exit1");
		buttons.add(b4);
		//b3.addActionListener(this);
		
		b5 = new JButton("Entry2");
		b5.setActionCommand("entry2");
		buttons.add(b5);
		//b3.addActionListener(this);
		
		b7 = new JButton("Egg3");
		b7.setActionCommand("egg3");
		buttons.add(b7);
		//b3.addActionListener(this);
		
		b8 = new JButton("Egg4");
		b8.setActionCommand("egg4");
		buttons.add(b8);
		//b3.addActionListener(this);
		
		b9 = new JButton("Exit2");
		b9.setActionCommand("exit2");
		buttons.add(b9);
		//b3.addActionListener(this);
		
		JPanel bp = new JPanel();
		bp.setSize(ep.getWidth(),200);
		bp.setPreferredSize(new Dimension(ep.getWidth(),200));
		//bp.setBackground(Color.YELLOW);
		bp.setLayout(new MigLayout("insets 8"));
		
		for (JButton b : buttons) {
			bp.add(b);
		}
		
		this.setLayout(new MigLayout("insets 0"));
		this.add(ep, "wrap");
		this.add(bp);
		
	    
		this.addWindowListener(
				new java.awt.event.WindowAdapter() {
					public void windowClosing(WindowEvent winEvt) {
						//fix
						close();
					}
				});
		
		//setup window
		this.setVisible(true);
		logger.info("Setting JFrame width to " + ep.calcWidth);
		this.setSize(ep.calcWidth, windowHeight);
		this.setPreferredSize(new Dimension(ep.calcWidth,windowHeight));
	}
	
	public void addButtonListener(ActionListener al){
		for (JButton b : buttons) {
			b.addActionListener(al);
		}
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
