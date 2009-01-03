package net.electroland.faces;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.util.Properties;
import java.util.StringTokenizer;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.electroland.artnet.ip.ArtNetDMXData;
import net.miginfocom.swing.MigLayout;

public class LAFACEBonusApp extends JFrame {

	// this quick and dirty code will enable you to do a few diagnostic things.

	// 1.) send "all on"
	// 2.) send "all off"
	// 3.) slide to an address in a universe, and specify the byte value for that address, plus the next two byte values.

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * 
	 */

	private JSlider universeSlider, channelSlider, val1Slider, val2Slider, val3Slider;
	private JLabel universe, channel, val1, val2, val3;
	private JButton send, all_off, all_on;
	private int recipientPort;
	private InetAddress recipient;

	public static final String DEFAULT_CONFIG_FNAME = "lights.properties";
	public static final String ALL_OFF = "All 00";
	public static final String ALL_ON = "All FF";
	public static final int ALL_FF_MODE = 1;
	public static final int ALL_00_MODE = 0;
	public static final int MANUAL_MODE = 2;
	public static final String SEND = "Send";
	public LAFACEBonusApp app;
	
	public LAFACEBonusApp(InetAddress recipient, int recipientPort){
		super("FACES Bonus App");
		
		this.app = this;
		this.recipient = recipient;
		this.recipientPort = recipientPort;
		
		System.out.println("will send to :" + recipient + " on port:" + recipientPort);
		
		this.setLayout(new MigLayout(""));

		universeSlider = new JSlider(0, 255, 0);
		universe = new JLabel("0");
		this.add(new JLabel("Universe:"));
		this.add(universeSlider);
		this.add(universe, "wrap");

		universeSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				universe.setText("" + universeSlider.getValue());
			}
		});
		
		
		channelSlider = new JSlider(1, 510, 1);
		channel = new JLabel("1");
		this.add(new JLabel("Channel:"));
		this.add(channelSlider);
		this.add(channel, "wrap");

		channelSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				channel.setText("" + channelSlider.getValue());
			}
		});
		
		
		val1Slider = new JSlider(0, 255, 0);
		val1 = new JLabel("00");
		this.add(new JLabel("Channel Val:"));
		this.add(val1Slider);
		this.add(val1, "wrap");

		val1Slider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				val1.setText(Util.bytesToHex(val1Slider.getValue()));
			}
		});
		
		
		val2Slider = new JSlider(0, 255, 0);
		val2 = new JLabel("00");
		this.add(new JLabel("Channel + 1 Val:"));
		this.add(val2Slider);
		this.add(val2, "wrap");

		val2Slider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				val2.setText(Util.bytesToHex(val2Slider.getValue()));
			}
		});
		
		
		val3Slider = new JSlider(0, 255, 0);
		val3 = new JLabel("00");
		this.add(new JLabel("Channel + 2 Val:"));
		this.add(val3Slider);
		this.add(val3, "wrap");

		val3Slider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				val3.setText(Util.bytesToHex(val3Slider.getValue()));
			}
		});

		// add "all on", "all off" and text input for the full packet.
		
		// missing: send IP address & port
		
		
		all_on = new JButton(ALL_ON);
		all_off = new JButton(ALL_OFF);
		send = new JButton(SEND);
		this.add(all_off);
		this.add(all_on);
		this.add(send, "wrap");

		send.addActionListener(new ActionListener(){
	        public void actionPerformed(ActionEvent e) {
	        	app.send(MANUAL_MODE);
	        }
		});

		all_off.addActionListener(new ActionListener(){
	        public void actionPerformed(ActionEvent e) {
	        	app.send(ALL_00_MODE);
	        }
		});
		
		
		all_on.addActionListener(new ActionListener(){
	        public void actionPerformed(ActionEvent e) {
	        	app.send(ALL_FF_MODE);
	        }
		});		
		
		/* activate close button for window. */
		this.addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(WindowEvent winEvt) {
		    	System.exit(0);
		    }
		});
		
		this.setSize(600, 400);
		this.setVisible(true);
	}
	
	private void send(int mode){

		try {
			DatagramSocket socket;
			socket = new DatagramSocket(app.recipientPort);

			int universe = app.universeSlider.getValue();
			int channel1 = app.channelSlider.getValue();
			int val1 = app.val1Slider.getValue();
			int val2 = app.val2Slider.getValue();
			int val3 = app.val3Slider.getValue();
			
			ArtNetDMXData dmx = new ArtNetDMXData();
			dmx.setUniverse((byte)universe);
	
			// we don't use these parts of the spec.
			dmx.setPhysical((byte)1);
			dmx.Sequence = (byte)0;	
	
			
			// set light data
			byte[] data = new byte[512];
			for (int i=0; i < 512; i++){
				switch (mode){
				case(ALL_FF_MODE):
					data[i] = (byte)255;
					break;
				case(ALL_00_MODE):
					data[i] = (byte)0;
					break;
				default:
					if (i == channel1 - 1){
						data[i] = (byte)val1;
					}else if (i == channel1){
						data[i] = (byte)val2;					
					}else if (i == channel1 + 1){
						data[i] = (byte)val3;					
					}else{
						data[i] = (byte)0;					
					}
					break;
				}
			}
			dmx.setData(data);
	
			ByteBuffer b = dmx.getBytes();
	
			System.out.println(Util.bytesToHex(b.array()));	
	
			DatagramPacket packet 	                
				= new DatagramPacket(b.array(), b.position(), app.recipient, app.recipientPort);

			socket.send(packet);
			socket.close();		
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	
	
	public static void main(String args[]){
		
		String filename = args.length == 0 ? DEFAULT_CONFIG_FNAME: args[0];

		Properties p = new Properties();
		try {
			p.load(new FileInputStream(new File(filename)));

			StringTokenizer st = new StringTokenizer(p.getProperty("bonusReceiver"), ", :");			
			new LAFACEBonusApp(InetAddress.getByName(st.nextToken()), Integer.parseInt(st.nextToken()));
			
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
}