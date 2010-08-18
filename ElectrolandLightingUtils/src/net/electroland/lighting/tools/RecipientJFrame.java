package net.electroland.lighting.tools;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.net.UnknownHostException;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextArea;

import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.tools.views.DetectorStates;
import net.electroland.util.OptionException;
import net.miginfocom.swing.MigLayout;

import org.apache.log4j.Logger;

public class RecipientJFrame extends JFrame implements ActionListener, KeyListener {
	
	static Logger logger = Logger.getLogger(RecipientJFrame.class);

	final public static String updateStr = "Update";
	final public static String revertStr = "Revert";
	final public static String onStr = "All On";
	final public static String offStr = "All Off";
	
	private Recipient recipient;
	private String backup;
	private AnimationManager am;
	private DetectorManager dm;
	private RecipientRepresentation representation;
	private JTextArea protocol;
	private JButton update, revert, off, on;
	private JComboBox display;
	private JLabel showName;

	public RecipientJFrame(Recipient recipient, AnimationManager am, DetectorManager dm)
	{
		this.recipient = recipient;
		this.backup = recipient.getOriginalStr();
		this.am = am;
		this.dm = dm;
		
		// set up frame
		update = new JButton(updateStr);	update.addActionListener(this);
		revert = new JButton(revertStr);	revert.addActionListener(this);
		on = new JButton(onStr);			update.addActionListener(this);
		off = new JButton(offStr);			revert.addActionListener(this);
		revert.setEnabled(false);

		protocol = new JTextArea(recipient.getOriginalStr());
		protocol.addKeyListener(this);

		String[] options = {"Current Animation", "Light States", 
							"Detector States", "Test Matrix"};
		display = new JComboBox(options);
		showName = new JLabel();
		display.setSelectedIndex(0);
		display.addActionListener(this);

		representation = new DetectorStates(recipient);

		// 6 columns
		this.setName(recipient.getID());
		this.setLayout(new MigLayout("wrap 6"));
		this.setSize(450, 500);

		// row 1
//		this.add(update);
//		this.add(revert);
//		this.add(new JLabel(), "span 4");// 4 blanks

		// row 2
		this.add(protocol, "span 6 3");
		
		// row 5
		this.add(on);
		this.add(off);
		this.add(new JLabel(), "span 2");// 2 blanks
		this.add(display);

		// row 6
		this.add(showName, "span 6");

		// row 7
		this.add(representation, "grow");
	}

	final public void update()
	{
		// re-parse the Recipient
		try {
			recipient = DetectorManager.parseRecipient(recipient.getID(), protocol.getText(), dm);

			// update the dm to remove existing Recipient objects (really, this should be in DM.)

			// notify parent to enable "save" button

			// disable revert, disable update
			update.setEnabled(false);
			revert.setEnabled(false);
		} catch (UnknownHostException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (OptionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}			
	}

	final public void revert()
	{
		// revert parse string
		recipient.setOriginalStr(backup);
		update();
	}

	final public void allOn()
	{
		recipient.allOff();
	}

	final public void allOff()
	{
		recipient.allOn();		
	}

	@Override
	public void keyPressed(KeyEvent arg0) {
		revert.setEnabled(true);
		update.setEnabled(true);
	}

	@Override
	public void keyReleased(KeyEvent arg0) {
		revert.setEnabled(true);
		update.setEnabled(true);
	}

	@Override
	public void keyTyped(KeyEvent arg0) {
		revert.setEnabled(true);
		update.setEnabled(true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		System.out.println(e);
		// set the view panel
		// tell AnimationManager to update the panel
		am.emptyRecipientRepresentationList();
		am.addRecipientRepresentation(representation);
	}
}