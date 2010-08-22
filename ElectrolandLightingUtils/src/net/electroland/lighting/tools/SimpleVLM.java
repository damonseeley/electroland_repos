package net.electroland.lighting.tools;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.Iterator;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JPanel;
import javax.swing.JRadioButtonMenuItem;

import net.electroland.lighting.conductor.Conductor;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.detector.models.BlueDetectionModel;
import net.electroland.lighting.detector.models.GreenDetectionModel;
import net.electroland.lighting.detector.models.RedDetectionModel;
import net.electroland.lighting.detector.models.ThresholdDetectionModel;
import net.electroland.lighting.tools.views.DetectorStates;

import org.apache.log4j.Logger;

public class SimpleVLM extends JFrame implements ActionListener, ItemListener{

	private static Logger logger = Logger.getLogger(SimpleVLM.class);

	private static final String ALL = "- All -";
	private static final String NONE = "- None -";
	private static final String RED = "RedDetectorModel";
	private static final String GREEN = "GreenDetectorModel";
	private static final String BLUE = "BlueDetectorModel";
	private static final String THRESH = "ThresholdDetectorModel";

	private DetectorManager dm;
	private AnimationManager am;
	private Conductor c;
	private JButton on, off, run, reload;
	private DetectorStates ds;
	private JMenu modelsMenu, recipsMenu;
	private String chosenModel;
	private String chosenRecip;
	
	// should pass in conductor to so that SystemStart is called.
	// (though, that would make it impossible NOT to use the conductor).
	// perhaps a "startable()" interface and a list of startables?

	public SimpleVLM(AnimationManager am, DetectorManager dm, Conductor c){
		this.c = c;
		this.dm = dm;
		this.am = am;
		init();
	}
	
	public SimpleVLM(AnimationManager am, DetectorManager dm)
	{
		this.dm = dm;
		this.am = am;
		init();
	}
	
	public void init()
	{		
		// just render the first recipient for now.
		Recipient first = dm.getRecipients().iterator().next();
		ds = new DetectorStates(first, true, null);
		am.addRecipientRepresentation(ds);
		this.add(ds, BorderLayout.CENTER);		
		
		JPanel controls = new JPanel();
        controls.setLayout(new FlowLayout());
        on = new JButton("All on");			on.addActionListener(this);
        off = new JButton("All off");		off.addActionListener(this);
        run = new JButton("Start");		run.addActionListener(this);
        reload = new JButton("Reload");		reload.addActionListener(this);
        reload.setEnabled(false);
        this.syncRunButton();

        
        ButtonGroup models = new ButtonGroup();
        modelsMenu = new JMenu("Detection Models");
 
        JRadioButtonMenuItem m1 = new JRadioButtonMenuItem(ALL);
        modelsMenu.add(m1);
        models.add(m1);

        m1.setSelected(true);
        chosenModel = m1.getText();
        
        m1.addItemListener(this);

        JRadioButtonMenuItem m2 = new JRadioButtonMenuItem(NONE);
        modelsMenu.add(m2);
        models.add(m2);
        m2.addItemListener(this);

        modelsMenu.addSeparator();

        JRadioButtonMenuItem m3 = new JRadioButtonMenuItem(RED);
        modelsMenu.add(m3);
        models.add(m3);
        m3.addItemListener(this);

        JRadioButtonMenuItem m4 = new JRadioButtonMenuItem(GREEN);
        modelsMenu.add(m4);
        models.add(m4);
        m4.addItemListener(this);

        JRadioButtonMenuItem m5 = new JRadioButtonMenuItem(BLUE);
        modelsMenu.add(m5);
        models.add(m5);
        m5.addItemListener(this);

        JRadioButtonMenuItem m6 = new JRadioButtonMenuItem(THRESH);
        modelsMenu.add(m6);
        models.add(m6);
        m6.addItemListener(this);
        
        ButtonGroup recips = new ButtonGroup();
        recipsMenu = new JMenu("Recipients");
		Iterator <Recipient> i = dm.getRecipients().iterator();
		boolean defaultSelected = false;
		while (i.hasNext())
		{
			JRadioButtonMenuItem recip = new JRadioButtonMenuItem(i.next().getID());
			recips.add(recip);
			recipsMenu.add(recip);
			if (!defaultSelected)
			{
				recip.setSelected(true);
				defaultSelected = true;
				chosenRecip = recip.getText();
			}
	        recip.addItemListener(this);
		}
		
        controls.add(on);
        controls.add(off);
        controls.add(run);
        controls.add(reload);

        JMenuBar menus = new JMenuBar();
        menus.add(recipsMenu);
        menus.add(modelsMenu);
        
        this.add(controls, BorderLayout.SOUTH);
        this.add(menus, BorderLayout.NORTH);
        this.setSize(450, 500);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setVisible(true);
	}

	public void actionPerformed(ActionEvent e) {
		if (e.getSource().equals(on))
		{
			dm.allOn();
			ds.repaint();
			
		}else if (e.getSource().equals(off))
		{
			dm.allOff();
			ds.repaint();

		}else if (e.getSource().equals(run))
		{
			if (am.isRunning()){
				// should be calling systemStop() in Conductor.
				if (c != null)
					c.stopSystem();
				else
					am.stop();
			}else{
				// should be calling systemStart() in Conductor.
				if (c != null)
					c.startSystem();
				else
					am.goLive();
				// set the latest recipient.
				ds.setRecipient(dm.getRecipient(chosenRecip));
			}
			syncRunButton();

		}else if (e.getSource().equals(reload))
		{
//			try {
//				dm.init(dm.getProps());
//				am.init(am.getFPS());
//			} catch (IOException f) {
//				logger.error(f);
//			} catch (OptionException f) {
//				logger.error(f);
//			}
		}
	}

	public void syncRunButton(){
		if (am.isRunning()){
			run.setText("Stop");
			on.setEnabled(false);
			off.setEnabled(false);
			reload.setEnabled(false);			
		}else{
			run.setText("Start");
			on.setEnabled(true);
			off.setEnabled(true);
			reload.setEnabled(false);
		}
	}

	private static boolean containedIn(JMenu menu, Component c)
	{
		Component mc[] = menu.getMenuComponents();
		for (int i = 0; i< mc.length; i++){
			if (mc[i].equals(c))
				return true;
		}
		return false;
	}

	public void itemStateChanged(ItemEvent e) {
		if (e.getStateChange() == ItemEvent.SELECTED)
		{
			if (e.getSource() instanceof JRadioButtonMenuItem){
				if (containedIn(recipsMenu, (Component)e.getSource())){
					chosenRecip = ((JRadioButtonMenuItem)e.getSource()).getText();
					ds.setRecipient(dm.getRecipient(chosenRecip));
					logger.info("Recipient: " + chosenRecip);
				}else if (containedIn(modelsMenu, (Component)e.getSource())){
					chosenModel = ((JRadioButtonMenuItem)e.getSource()).getText();
					
					if (ALL.equals(chosenModel)){
						ds.showAllModels();
						ds.setShowDetectors(true);
					}else if (NONE.equals(chosenModel)){
						ds.setShowDetectors(false);
					}else if (RED.equals(chosenModel)){
						ds.limitDisplayToModel(new RedDetectionModel());
						ds.setShowDetectors(true);
					}else if (GREEN.equals(chosenModel)){
						ds.limitDisplayToModel(new GreenDetectionModel());
						ds.setShowDetectors(true);
					}else if (BLUE.equals(chosenModel)){
						ds.limitDisplayToModel(new BlueDetectionModel());
						ds.setShowDetectors(true);
					}else if (THRESH.equals(chosenModel)){
						ds.limitDisplayToModel(new ThresholdDetectionModel());
						ds.setShowDetectors(true);
					}
					
					ds.repaint();
					
					logger.info("Model: " + chosenModel);
				}
			}
		}
	}

	
}