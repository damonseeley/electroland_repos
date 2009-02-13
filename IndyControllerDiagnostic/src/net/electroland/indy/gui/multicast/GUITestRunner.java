package net.electroland.indy.gui.multicast;

//comment
//comment 2


import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.net.UnknownHostException;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JRadioButton;
import javax.swing.JSeparator;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.electroland.indy.test.IPAddressParseException;
import net.electroland.indy.test.ReceiverThread;
import net.electroland.indy.test.Target;
import net.electroland.indy.test.Util;
import net.miginfocom.swing.MigLayout;

@SuppressWarnings("serial")
public class GUITestRunner extends JFrame {

	protected JSlider byteSlider, fpsSlider, cmdSlider;	
	protected JButton oneButton, streamButton;
	protected JRadioButton triangle, slider;
	protected ButtonGroup buttonGroup;
	protected JLabel fps, cmdByte, byteVal, fpsVal;
	protected JCheckBox oddsCompliment;
	private SliderThread sender;
	private ReceiverThread[] receivers;
	private Target[] targets;
	private final String START_STREAM = "Start stream";
	private final String STOP_STREAM = "Stop stream";
	private GUITestRunner runner;
	
	public GUITestRunner(int width, int height){
		
		super("Lantronix Multi-port testing");

		this.setLayout(new MigLayout(""));

		this.add(new JLabel("Cmd Byte:"), "gap 10");
		cmdByte = new JLabel("01");
		this.add(cmdByte, "span, growx, wrap");

		cmdSlider = new JSlider(0, 253, 1);
		this.add(cmdSlider, "span, growx, wrap");
		cmdSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				byte[] b = {(byte)cmdSlider.getValue()};
				cmdByte.setText(Util.bytesToHex(b));
			}
		});		
		
		// ---------------------------------------------
		this.add(new JSeparator(), "span, growx, wrap");
		
		this.add(new JLabel("Data Bytes:"), "gap 10");
		byteVal = new JLabel("01");
		this.add(byteVal, "span, growx, wrap");

		byteSlider = new JSlider(1, 253, 1);
		this.add(byteSlider, "span, growx, wrap");
		byteSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				byte[] b = {(byte)byteSlider.getValue()};
				byteVal.setText(Util.bytesToHex(b));
			}
		});

		// ---------------------------------------------
		this.add(new JSeparator(), "span, growx, wrap");

		this.add(new JLabel("Frames per second:"), "gap 10");
		fpsVal = new JLabel("30");
		this.add(fpsVal, "span, growx, wrap");

		fpsSlider = new JSlider(1, 100, 30);
		this.add(fpsSlider, "span, growx, wrap");
		fpsSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				fpsVal.setText("" + fpsSlider.getValue());
			}
		});

		this.add(new JLabel("Measured FPS:"), "gap 10");
		fps = new JLabel("");
		this.add(fps, "span, growx, wrap");

		this.add(new JSeparator(), "span, growx, wrap");	

		buttonGroup = new ButtonGroup();
		
		triangle = new JRadioButton("Triangle Wave", false);
		this.add(triangle, "growx");
		buttonGroup.add(triangle);		

		slider = new JRadioButton("Slider Value", true);
		this.add(slider, "span, growx");
		buttonGroup.add(slider);

		// ---------------------------------------------
		this.add(new JSeparator(), "span, growx, wrap");		

		oddsCompliment = new JCheckBox("Odd bytes are complimentary", true);
		this.add(oddsCompliment, "center, span 2, wrap");
		
		
		// ---------------------------------------------
		this.add(new JSeparator(), "span, growx, wrap");		

		oneButton = new JButton("Send one");
		this.add(oneButton, "center");
		streamButton = new JButton(START_STREAM);
		this.add(streamButton, "center, span 2, wrap");
		
		runner = this;

		oneButton.addActionListener(new ActionListener(){
	        public void actionPerformed(ActionEvent e) {
	        	if (oneButton.isEnabled()){
	        		sender = new SliderThread(targets, runner, true);
		        	sender.start();	        		
	        	}
	        }
		});
		
		
		streamButton.addActionListener(new ActionListener(){
	        public void actionPerformed(ActionEvent e) {
	        	if (streamButton.getText() == START_STREAM) {
	        		oneButton.setEnabled(false);
	        		streamButton.setText(STOP_STREAM);
	        		sender = new SliderThread(targets, runner, false);
	        		sender.start();
	        	} else {
	        		streamButton.setText(START_STREAM);
	        		sender.stopClean();
	        		sender = null;
	        		fps.setText("");
	        		oneButton.setEnabled(true);
	        	}
	        }
		});

		this.setVisible(true);
		this.setSize(width, height);

		this.addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(WindowEvent winEvt) {
		    	close();
		    }
		});
	}
	
//	@SuppressWarnings("deprecation")
	private void close(){
		if (sender != null){
			sender.stop();			
		}
		for (int i = 0; i < receivers.length; i++){
			receivers[i].stop();
		}
		System.exit(0);
	}

	private void init(String args[]){
		try{

			// rest of the args are ipaddres:port
			targets = new Target[args.length];
			for (int i = 0; i < args.length; i++){
				targets[i] = new Target(args[i]);
			}

			// create receiver threads
			receivers = new ReceiverThread[targets.length];
			for (int i=0; i < targets.length; i++ ){
				receivers[i] = new ReceiverThread(targets[i]);
				receivers[i].start();
			}			

		}catch(NumberFormatException e){
			e.printStackTrace();			
			System.out.println("usage: java TestRunner [fps] [host1:port1]...");
		}catch (UnknownHostException e) {
			e.printStackTrace();
			System.out.println("usage: java TestRunner [fps] [host1:port1]...");
		} catch (IPAddressParseException e) {
			e.printStackTrace();
			System.out.println("usage: java TestRunner [fps] [host1:port1]...");
		}		
	}
	
	public static void main(String args[]){		
		new GUITestRunner(300, 400).init(args);		
	}
}