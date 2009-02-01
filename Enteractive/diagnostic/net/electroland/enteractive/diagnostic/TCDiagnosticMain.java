package net.electroland.enteractive.diagnostic;

import java.awt.Dimension;
import java.awt.event.WindowEvent;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

import javax.swing.BorderFactory;
import javax.swing.JFrame;

import net.electroland.enteractive.diagnostic.old.LTUDPBroadcasterManager;
import net.electroland.enteractive.udpUtils.UDPReceiver;
import net.miginfocom.swing.MigLayout;
import org.apache.log4j.Logger;


public class TCDiagnosticMain extends JFrame {
	
	static Logger logger = Logger.getLogger(TCDiagnosticMain.class);
	
	private TCAddressPanel addressPanel;
	private TCPacketPanel packetPanel;
	private TCOutputPanel outputPanel;
	
	private TCUDPReceiver udpr;
	private TCBroadcaster tcb;
	
	//private int initSendPort;
	private int rcvPort;
	
	public static HashMap<String, String> properties = new HashMap<String, String>();
	BufferedReader input;
		
	public TCDiagnosticMain(int w, int h) {
		super("Electroland Tile Controller Diagnostic");
		
		loadProperties();
		
		//initSendPort = 10000;
		rcvPort = 10001;
		
		try {
			udpr = new TCUDPReceiver(rcvPort);
			udpr.start();
		} catch (Exception e) {
		}
		
		//init broadcaster with some dummy args
		tcb = new TCBroadcaster(TCDiagnosticMain.properties.get("IPAddress"), Integer.parseInt(TCDiagnosticMain.properties.get("sendPort")));
		
		int inset = 10;
		
		MigLayout layout = new MigLayout(
				"inset "+ inset, // Layout Constraints
				"[]", // Column constraints
				"5[pref!]5[pref!]5[]10"); // Row constraints
		
		setLayout(layout);
		
		
		addressPanel = new TCAddressPanel(inset, TCDiagnosticMain.properties.get("IPAddress"),  Integer.parseInt(TCDiagnosticMain.properties.get("sendPort")), tcb);
		addressPanel.setMinimumSize(new Dimension (w-inset*2, inset));
		addressPanel.setBorder(BorderFactory.createTitledBorder("IP Address Setup"));
		
		packetPanel = new TCPacketPanel(inset, tcb);
		packetPanel.setMinimumSize(new Dimension (w-inset*2, inset));
		packetPanel.setBorder(BorderFactory.createTitledBorder("Packets"));
		
		outputPanel = new TCOutputPanel(inset);
		outputPanel.setMinimumSize(new Dimension (w-inset*2, inset));
		outputPanel.setBorder(BorderFactory.createTitledBorder("UDP Return"));
		
		add(addressPanel,"wrap");
		add(packetPanel,"wrap");
		add(outputPanel);
		
		udpr.registerOutput(outputPanel);

		//Dimension packetPanelSize = packetPanel.getPreferredSize();
		//packetPanel.setBounds(0,0, (int) packetPanelSize.getWidth(),h/2);
		
		setVisible(true);
		//setPreferredSize(new Dimension(w, h));
		setSize(w,h);
		
		addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(WindowEvent winEvt) {
		    	close();
		    }
		});

	}
	
	public void loadProperties(){
		String line;
		String[] items;
		
		try{
			input = new BufferedReader(new FileReader("depends/diagnosticProperties.txt"));
		} catch (FileNotFoundException e){
			logger.error(e.getMessage(), e);
		}
		try{
			while((line = input.readLine()) != null){
				if(!line.startsWith("#") && line.length() > 0){	// if line is not a comment or blank...
					logger.info(line);
					items = line.split("=");					// split variable and value
					properties.put(items[0].trim(), items[1].trim());	// add to properties table
				}
			}
		} catch (IOException e){
			logger.error(e.getMessage(), e);
		}
	}

	public void close() {
		logger.info("Shutting Down");
		try {
			Thread.sleep(100);
		} catch (InterruptedException e) {
		} // wait a few second in case threads need to shut down nicely
		try {
			udpr.stopRunning();
		} catch (Exception e) {
			// TODO: handle exception
			logger.info("UDP Logger failed with error " + e);
		}
		System.exit(0);
	}

	
	public static void main(String[] args) {
		new TCDiagnosticMain(600,700);
	}

}
