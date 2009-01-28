package net.electroland.indy.gui.multicast;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Properties;
import java.util.StringTokenizer;

import javax.swing.BorderFactory;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSeparator;
import javax.swing.JSlider;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.electroland.indy.test.IPAddressParseException;
import net.electroland.indy.test.Util;
import net.miginfocom.swing.MigLayout;

/**
 * Crazy GUI tester.  This thing is tightly coupled with Sender2Thread and
 * Receiver2Thread.  Ideally, it should implement a Loggable interface to pass
 * to those objects, and an Updateable interface to update and query the GUI for
 * values.
 * 
 * A couple nice updates to have:
 * 1.) if you are streaming custom packets, disable the text input for custom packets.
 * 2.) enable all default values to be specified in the properties file
 * 3.) enable a multicastTarget property that starts a multicast thread.
 * 4.) when the offset broadcast frame is up, send the offset broadcast AND
 *     the data packets, so you don't drop a frame.
 * 5.) disable the offset delay if "intersperse offsets" isn't enabled.  probably
 *     move "intersperse offsets" to the "Broadcast packets" panel.
 * 6.) disable "Include time" if Data packet length <= 13.
 * 7.) disable "Data bytes" slider if "Slider Value" is disabled.
 * 8.) disable "Odd bytes are complimentary" if "Custom packet", "Oscillating
 *     packets", "Step through recipients" are enabled.
 * 9.) group the options for which odd bytes can be complimentary.
 * 10.) report the empirical seconds between offset broadcasts.
 *
 * When TCP is enabled, the following must be disabled:
 * 
 * 1.) cmd byte
 * 2.) offset delay
 * 3.) stop through recipients (temp)
 * 
 * 
 * 
 * @author geilfuss
 *
 */

@SuppressWarnings("serial")
public class MulticastRunner extends JFrame {

	protected JSlider byteSlider, fpsSlider, cmdSlider, 
						offsetBytesSlider, pcktLengthSlider, offsetDelay;	
	protected JButton oneButton, streamButton, startByteButton;
	protected JRadioButton triangle, slider, ascending, custom, 
							oscillating, stepThroughRecipients;
	protected JTextField customPacket;
	protected ButtonGroup buttonGroup;
	protected JLabel fps, cmdByte, byteVal, fpsVal, offsetBytes, pcktLengthVal, 
						offsetDelayVal;
	protected JCheckBox oddsCompliment, includeTimeing, includeOffset,
						logSends, logOffsets, logReceives, logTimes, useTCP;
	private Sender2Thread sender;
	private TCPSenderThread tcpsender;
	private Receiver2Thread[] receivers;
	private Target2[] udpTargets;
	private Target2[] tcpTargets;
	public final static String START_STREAM = "Start stream";
	public final static String STOP_STREAM = "Stop stream";
	private MulticastRunner runner;
	private int seedCtr = 0;

	public MulticastRunner(int width, int height){
		
		super("Lantronix Multi-port testing");
		
		JPanel multicastPanel = new JPanel();
		multicastPanel.setBorder(BorderFactory.createTitledBorder("Test packets"));
		this.setLayout(new MigLayout(""));
		this.add(multicastPanel, "center, growx, wrap");
		// add everhting below to the above.  add the above to this.
		
		multicastPanel.setLayout(new MigLayout(""));

		multicastPanel.add(new JLabel("Cmd Byte:"), "gap 10");
		cmdByte = new JLabel("00");
		multicastPanel.add(cmdByte, "span 2, growx, wrap");

		cmdSlider = new JSlider(0, 253, 0);
		multicastPanel.add(cmdSlider, "span, growx, wrap");
		cmdSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				byte[] b = {(byte)cmdSlider.getValue()};
				cmdByte.setText(Util.bytesToHex(b));
			}
		});		
		
		// ---------------------------------------------
		multicastPanel.add(new JSeparator(), "span, growx, wrap");
		
		multicastPanel.add(new JLabel("Data Bytes:"), "gap 10");
		byteVal = new JLabel("00");
		multicastPanel.add(byteVal, "span 2, growx, wrap");

		byteSlider = new JSlider(0, 253, 0);
		multicastPanel.add(byteSlider, "span, growx, wrap");
		byteSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				byte[] b = {(byte)byteSlider.getValue()};
				byteVal.setText(Util.bytesToHex(b));
			}
		});

		// ---------------------------------------------
		multicastPanel.add(new JSeparator(), "span, growx, wrap");
		
		multicastPanel.add(new JLabel("Data packet length:"), "gap 10");
		pcktLengthVal = new JLabel("12");
		multicastPanel.add(pcktLengthVal, "span 2, growx, wrap");

		pcktLengthSlider = new JSlider(0, 508, 12);
		multicastPanel.add(pcktLengthSlider, "span, growx, wrap");
		pcktLengthSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				pcktLengthVal.setText("" + pcktLengthSlider.getValue());
			}
		});		
		
		// ---------------------------------------------
		multicastPanel.add(new JSeparator(), "span, growx, wrap");

		multicastPanel.add(new JLabel("Frames per second:"), "gap 10");
		fpsVal = new JLabel("30");
		multicastPanel.add(fpsVal, "span 2, growx, wrap");

		fpsSlider = new JSlider(1, 100, 30);
		multicastPanel.add(fpsSlider, "span, growx, wrap");
		fpsSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				fpsVal.setText("" + fpsSlider.getValue());
			}
		});

		multicastPanel.add(new JLabel("Measured FPS:"), "gap 10");
		fps = new JLabel("");
		multicastPanel.add(fps, "span 2, growx, wrap");

		/* --------------------- */
		multicastPanel.add(new JSeparator(), "span, growx, wrap");	

		buttonGroup = new ButtonGroup();

		slider = new JRadioButton("Slider Value", true);
		multicastPanel.add(slider, "growx");
		buttonGroup.add(slider);

		triangle = new JRadioButton("Triangle Wave", false);
		multicastPanel.add(triangle, "growx");
		buttonGroup.add(triangle);		
		
		ascending = new JRadioButton("Ascending bytes", false);
		multicastPanel.add(ascending, "growx, wrap");
		buttonGroup.add(ascending);

		oscillating = new JRadioButton("Oscillating packets", false);
		multicastPanel.add(oscillating, "growx");
		buttonGroup.add(oscillating);

		stepThroughRecipients = new JRadioButton("Step through recipients (trace pattern)", false);
		multicastPanel.add(stepThroughRecipients, "span 2, wrap");
		buttonGroup.add(stepThroughRecipients);		
		
		custom = new JRadioButton("Custom packet: ", false);
		multicastPanel.add(custom, "growx");
		buttonGroup.add(custom);
		
		customPacket = new JTextField();
		multicastPanel.add(customPacket, "span2 ,growx, wrap");
		
		// ---------------------------------------------
		multicastPanel.add(new JSeparator(), "span, growx, wrap");		

		includeOffset = new JCheckBox("Intersperse offsets", false);
		multicastPanel.add(includeOffset, "span 2, left");
		
		oddsCompliment = new JCheckBox("Odd bytes are complimentary", false);
		multicastPanel.add(oddsCompliment, "left, wrap");

		includeTimeing = new JCheckBox("Include time", false);
		multicastPanel.add(includeTimeing, "span 2, left");

		useTCP = new JCheckBox("Use TCP", false);
		multicastPanel.add(useTCP, "left, wrap");
		
		// ---------------------------------------------
		multicastPanel.add(new JSeparator(), "span, growx, wrap");		

		oneButton = new JButton("Send one");
		multicastPanel.add(oneButton, "center");
		streamButton = new JButton(START_STREAM);
		multicastPanel.add(streamButton, "center");
		startByteButton = new JButton("Broadcast offsets");
		multicastPanel.add(startByteButton, "center, wrap");
		
		runner = this;

		oneButton.addActionListener(new ActionListener(){
	        public void actionPerformed(ActionEvent e) {
	        	if (oneButton.isEnabled()){
	        		if (useTCP.isSelected()){
        				tcpsender= new TCPSenderThread(runner, tcpTargets);
        				tcpsender.start();
	        		}else{
		        		sender = new Sender2Thread(udpTargets, runner, true, seedCtr++);
			        	sender.start();	        		
	        		}
	        	}
	        }
		});
		

		streamButton.addActionListener(new ActionListener(){
	        public void actionPerformed(ActionEvent e) {
	        	if (streamButton.getText() == START_STREAM) {
	        		oneButton.setEnabled(false);
	        		startByteButton.setEnabled(false);
	        		streamButton.setText(STOP_STREAM);
	        		if (useTCP.isSelected()){
        				tcpsender= new TCPSenderThread(runner, tcpTargets);
        				tcpsender.start();
	        		}else{
		        		sender = new Sender2Thread(udpTargets, runner, false);
		        		sender.start();	        			
	        		}
	        	} else {
	        		if (useTCP.isSelected()){
        				tcpsender.stopClean();
        				tcpsender = null;
	        		}else{
		        		sender.stopClean();
		        		sender = null;
	        		}
	        		streamButton.setText(START_STREAM);
	        		fps.setText("");
	        		oneButton.setEnabled(true);
	        		startByteButton.setEnabled(true);
	        	}
	        }
		});

		startByteButton.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
        		sender = new Sender2Thread(udpTargets, runner, true, -1);
	        	sender.start();				
			}
		});
		
		this.setVisible(true);
		this.setSize(width, height);

		this.addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(WindowEvent winEvt) {
		    	close();
		    }
		});
		
		// --- offset packets
		JPanel offsetPanel = new JPanel();
		offsetPanel.setLayout(new MigLayout(""));
		offsetPanel.setBorder(BorderFactory.createTitledBorder("Offset packets"));
		this.add(offsetPanel, "span, growx, wrap");

		
		offsetPanel.add(new JLabel("Offset delay (frames):"), "gap 10");

		offsetDelay = new JSlider(1, 300, 30);
		offsetPanel.add(offsetDelay, "span 5, width 300, growx");
		offsetDelay.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				offsetDelayVal.setText("" + offsetDelay.getValue());
			}
		});		

		offsetDelayVal = new JLabel("30");
		offsetPanel.add(offsetDelayVal, "wrap");

		
		// --- logging
		JPanel loggingPanel = new JPanel();
		loggingPanel.setLayout(new MigLayout(""));
		loggingPanel.setBorder(BorderFactory.createTitledBorder("Logging options"));
		this.add(loggingPanel, "span, growx, wrap");
		
		logSends = new JCheckBox("Data packets sent", true);
		loggingPanel.add(logSends, "left, span 2, growx");

		logOffsets = new JCheckBox("Offset packets sent", true);
		loggingPanel.add(logOffsets, "wrap");

		logReceives = new JCheckBox("Received packets", true);
		loggingPanel.add(logReceives, "left, span 2");

		logTimes = new JCheckBox("Round trip durations", true);
		loggingPanel.add(logTimes, "wrap");
		
	}

	//	@SuppressWarnings("deprecation")
	private void close(){
		if (sender != null){
			sender.stopClean();
		}
		if (tcpsender != null){
			tcpsender.stopClean();
		}
		for (int i = 0; i < receivers.length; i++){
			receivers[i].stopClean();
		}
		System.exit(0);
	}

	/**
	 * Loads the multicast.properties file (or a similar file, specified by the
	 * first and only command line arg) and parses the data into senders and
	 * receivers.
	 * 
	 * @param args
	 */
	@SuppressWarnings("unchecked")
	private void init(String args[]){
		try{
			String filename = args.length == 0 ? "multicast.properties": args[0];

			Properties p = new Properties();
			p.load(new FileInputStream(new File(filename)));

			StringTokenizer targetTokenizer
				= new StringTokenizer(p.getProperty("UDPTargets")," ,\t");

			ArrayList temp = new ArrayList();
			
			// parse and create UDP senders.
			while (targetTokenizer.hasMoreTokens()){

				StringTokenizer st = new StringTokenizer(targetTokenizer.nextToken(),":");				
				temp.add(new Target2(st.nextToken(), 
									Integer.parseInt(st.nextToken()),
											Integer.parseInt(st.nextToken())));
			}

			udpTargets = new Target2[temp.size()];
			temp.toArray(udpTargets);

			// parse TCP senders
			targetTokenizer
				= new StringTokenizer(p.getProperty("TCPTargets")," ,\t");

			temp.clear();
			
			while (targetTokenizer.hasMoreTokens()){

				StringTokenizer st = new StringTokenizer(targetTokenizer.nextToken(),":");				
				temp.add(new Target2(st.nextToken(), 
									Integer.parseInt(st.nextToken())));
			}

			tcpTargets = new Target2[temp.size()];
			temp.toArray(tcpTargets);
			
			// parse UDP response listener ports
			StringTokenizer portTokenizer
				= new StringTokenizer(p.getProperty("UDPResponsePorts")," ,\t");

			temp.clear();
			while (portTokenizer.hasMoreTokens()){
				temp.add(new Integer(portTokenizer.nextToken()));
			}

			// create listeners and start them up.
			receivers = new Receiver2Thread[temp.size()];
			for (int i=0; i < temp.size(); i++ ){
				receivers[i] = new Receiver2Thread(((Integer)temp.get(i)).intValue(), this);
				receivers[i].start();
			}
			
			

		}catch(NumberFormatException e){
			e.printStackTrace();
			System.out.println("usage: java TestRunner [config file]");
		}catch (UnknownHostException e) {
			e.printStackTrace();
			System.out.println("usage: java TestRunner [config file]");
		} catch (IPAddressParseException e) {
			e.printStackTrace();
			System.out.println("usage: java TestRunner [config file]");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.out.println("usage: java TestRunner [config file]");
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("usage: java TestRunner [config file]");
		}
	}
	public static void main(String args[]){		
		new MulticastRunner(575, 830).init(args);
	}
}