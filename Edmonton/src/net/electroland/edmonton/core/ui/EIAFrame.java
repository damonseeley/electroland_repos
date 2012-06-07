package net.electroland.edmonton.core.ui;

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
import java.util.Map;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.electroland.edmonton.core.SoundController;
import net.miginfocom.swing.MigLayout;

import org.apache.log4j.Logger;

@SuppressWarnings("serial")
public class EIAFrame extends JFrame implements ActionListener {

	EIATiledPanel ep;

	private int windowHeight;
	public Map<String, Object> context;

	private JButton testShow,showHideGfx,pm1avg;
	private JLabel dScaleLabel;
	private JPanel bp;
	//private DoubleJSlider jsScale;
	private ArrayList<JButton> buttons;

	static Logger logger = Logger.getLogger(EIAFrame.class);

	public EIAFrame(int width, int height, Map<String,Object> context) {

		super("Electroland @ EIA");

		this.context = context;

		windowHeight = height;

		ep = new EIATiledPanel(context);

		//Font newButtonFont = new Font("Default",Font.PLAIN,13);  

		buttons = new ArrayList<JButton>();

		showHideGfx = new JButton("Hide Graphics");
		showHideGfx.setActionCommand("showHideGfx");
		buttons.add(showHideGfx);
		        
        testShow = new JButton("Test Show");
        testShow.setActionCommand("testShow");
        buttons.add(testShow);
        
        pm1avg = new JButton("Log PM1 Average");
        pm1avg.setActionCommand("pm1avg");
        buttons.add(pm1avg);


		/**
		 * button panel
		 */
		bp = new JPanel();
		bp.setSize(ep.getWidth(),50);
		bp.setPreferredSize(new Dimension(ep.getWidth(),50));
		bp.setLayout(new MigLayout("insets 8"));

		for (JButton b : buttons) {
			bp.add(b);
		}



		/**
		 * slider to affect displayScale
		 */
		int scaleFactor = 100;
		int scaleMax = 10;
		final DoubleJSlider jsScale = new DoubleJSlider(0, scaleMax*scaleFactor, 0, scaleFactor);
		jsScale.setValue((int)(ep.getDisplayScale()*scaleFactor));
		jsScale.setMinimum(scaleFactor);

		//Create the label table
		Hashtable<Integer,JLabel> labelTable = new Hashtable<Integer,JLabel>();
		labelTable.put( new Integer(scaleFactor), new JLabel("1.0") );
		labelTable.put( new Integer(scaleMax*scaleFactor), new JLabel("10.0") );
		jsScale.setLabelTable( labelTable );
		jsScale.setPaintLabels(true);

		jsScale.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) {
				logger.info(jsScale.getScaledValue());
				// change ep displayScale
				ep.setDisplayScale(jsScale.getScaledValue());
				dScaleLabel.setText("Scale: " + jsScale.getValue()/100.0);
				setSize();
			}
		});

		dScaleLabel = new JLabel("Scale: " + jsScale.getValue()/100.0, JLabel.CENTER);
		dScaleLabel.setAlignmentX(Component.CENTER_ALIGNMENT);



		/**
		 * slider to affect sequencer delay
		 */

		//Create the label table
		Hashtable<Integer,JLabel> labelTable2 = new Hashtable<Integer,JLabel>();
		labelTable2.put( -1000, new JLabel("-1000") );
		labelTable2.put( 1000, new JLabel("1000") );

		/**
		 * add sliders
		 */
		JPanel sp = new JPanel();
		sp.setSize(ep.getWidth(),50);
		sp.setPreferredSize(new Dimension(ep.getWidth(),50));
		sp.setLayout(new MigLayout("insets 16"));

		sp.add(jsScale);
		sp.add(dScaleLabel, "wrap");

		/**
		 * put it all together (add it)
		 */
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
		logger.info("Close called on EIAFrame");
		SoundController sc = (SoundController)context.get("soundController");
		sc.shutdown();
		System.exit(0);
	}

	public void showHideGfx() {
		boolean state = ep.showHideGfx();
		if (state){
			logger.info("graphics turned on");
			showHideGfx.setText("Hide Graphics");
		} else {
			logger.info("graphics turned off");
			showHideGfx.setText("Show Graphics");
		}
	}



}

@SuppressWarnings("serial")
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