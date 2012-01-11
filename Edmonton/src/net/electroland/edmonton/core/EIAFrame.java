package net.electroland.edmonton.core;

/**
 * By Damon Seeley
 * Should contain all UI elements for the EIA app in one place
 * except for the main image area, drawn in EIAPanel
 * 
 */

import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.util.ArrayList;
import java.util.Hashtable;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.miginfocom.swing.MigLayout;

import org.apache.log4j.Logger;

@SuppressWarnings("serial")
public class EIAFrame extends JFrame implements ActionListener {

	EIATiledPanel ep;

	private int windowWidth,windowHeight;
	public Hashtable<String, Object> context;

	private int panelWidth,panelHeight;

	private JButton b1,b2,b3,b4,b5,b6,b7,b8,b9,b10;
	private JButton startSeq,stopSeq;
	private DoubleJSlider js1;
	private ArrayList<JButton> buttons;

	static Logger logger = Logger.getLogger(EIAFrame.class);

	public EIAFrame(int width, int height, Hashtable context) {

		super("Electroland @ EIA");

		this.context = context;

		windowWidth = width;
		windowHeight = height;

		ep = new EIATiledPanel(context);

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
		
		startSeq = new JButton("Start Sequence");
		startSeq.setActionCommand("startSeq");
		buttons.add(startSeq);

		stopSeq = new JButton("Stop Active Sequence");
		stopSeq.setActionCommand("stopSeq");
		buttons.add(stopSeq);
		
		


		// displayScale slider

		JLabel dsLabel = new JLabel("Scale", JLabel.CENTER);
		dsLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

		int scaleFactor = 100;
		int scaleMax = 10;
		final DoubleJSlider js1 = new DoubleJSlider(0, scaleMax*scaleFactor, 0, scaleFactor);
		js1.setValue((int)(ep.getDisplayScale()*scaleFactor));
		js1.setMinimum(scaleFactor);

		//Create the label table
		Hashtable<Integer,JLabel> labelTable = new Hashtable<Integer,JLabel>();
		labelTable.put( new Integer(scaleFactor), new JLabel("1.0") );
		labelTable.put( new Integer(scaleMax*scaleFactor), new JLabel("10.0") );
		js1.setLabelTable( labelTable );
		js1.setPaintLabels(true);

		js1.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) {
				logger.info(js1.getScaledValue());
				// change ep displayScale
				ep.setDisplayScale(js1.getScaledValue());
				setSize();
			}
		});


		JPanel bp = new JPanel();
		bp.setSize(ep.getWidth(),50);
		bp.setPreferredSize(new Dimension(ep.getWidth(),50));
		bp.setLayout(new MigLayout("insets 8"));

		for (JButton b : buttons) {
			bp.add(b);
		}

		JPanel sp = new JPanel();
		sp.setSize(ep.getWidth(),50);
		sp.setPreferredSize(new Dimension(ep.getWidth(),50));
		sp.setLayout(new MigLayout("insets 16"));
		sp.add(dsLabel);
		sp.add(js1);

		this.setLayout(new MigLayout("insets 0"));
		this.add(ep, "wrap");
		this.add(bp, "wrap");
		this.add(sp);



		this.addWindowListener(
				new java.awt.event.WindowAdapter() {
					public void windowClosing(WindowEvent winEvt) {
						//fix
						close();
					}
				});

		//setup window
		this.setVisible(true);
		//ogger.info("Setting JFrame width to " + ep.calcWidth);
		setSize();
	}

	public void setSize(){
		this.setSize(ep.getPanelWidth(), windowHeight);
		this.setPreferredSize(new Dimension(ep.getPanelWidth(),windowHeight));
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

		//logger.info(e.getActionCommand());

	}


	private void close(){
		SoundController sc = (SoundController)context.get("soundcontroller");
		//sc.shutdown();
		System.exit(0);
	}



}

class DoubleJSlider extends JSlider {

	final int scale;

	public DoubleJSlider(int min, int max, int value, int scale) {
		super(min, max, value);
		this.scale = scale;
	}

	public double getScaledValue() {
		return ((double)super.getValue()) / this.scale;
	}
}


