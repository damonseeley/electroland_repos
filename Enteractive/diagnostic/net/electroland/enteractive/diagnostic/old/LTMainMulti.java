package net.electroland.enteractive.diagnostic.old;

import java.awt.Dimension;
import java.awt.event.WindowEvent;

import javax.swing.BorderFactory;
import javax.swing.JFrame;

import net.electroland.enteractive.udpUtils.UDPReceiver;
import net.miginfocom.swing.MigLayout;


public class LTMainMulti extends JFrame {
	
	LTPacketPanel packetPanel;
	LTButtonPanel buttonPanel;
	LTOutputPanel outputPanel;
	
	public LTUDPReceiver udpr;
	//public LTUDPLogger udpr;
	private int sendPort;
	private int rcvPort;
	
	private LTUDPBroadcasterManager lM;
	
	public LTMainMulti(int w, int h) {
		super("Lantronix Testing");
		
		sendPort = 10001;
		rcvPort = 10001;
		
		try {
			udpr = new LTUDPReceiver(rcvPort);
			//udpr = new LTUDPLogger("UDPLog.txt", rcvPort);
			udpr.start();
		} catch (Exception e) {
		}
		
		String startIP = "192.168.0.11";
		
		//int numLTUBS, int startPort, int startIP
		lM = new LTUDPBroadcasterManager(1,sendPort,11);

		
		MigLayout layout = new MigLayout(
				"inset 10", // Layout Constraints
				"[]", // Column constraints
				"10[pref!]20[pref!]20[]10"); // Row constraints
		
		setLayout(layout);
		//setPreferredSize(new Dimension(w, h));

		int inset = 10;
		
		packetPanel = new LTPacketPanel(inset, startIP, lM);
		packetPanel.setMinimumSize(new Dimension (w-inset*2, 10));
		packetPanel.setBorder(BorderFactory.createTitledBorder("Packet Setup"));

		buttonPanel = new LTButtonPanel(inset, lM);
		buttonPanel.setMinimumSize(new Dimension (w-inset*2, 10));
		buttonPanel.setBorder(BorderFactory.createTitledBorder("Comm Control"));
		
		outputPanel = new LTOutputPanel(inset);
		outputPanel.setMinimumSize(new Dimension (w-inset*2, 10));
		outputPanel.setBorder(BorderFactory.createTitledBorder("UDP Return"));
		
		add(packetPanel,"wrap");
		add(buttonPanel,"wrap");
		add(outputPanel);
		
		udpr.registerOutput(outputPanel);

		//Dimension packetPanelSize = packetPanel.getPreferredSize();
		//packetPanel.setBounds(0,0, (int) packetPanelSize.getWidth(),h/2);
		
		setVisible(true);
		setSize(w,h);

		
		addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(WindowEvent winEvt) {
		    	close();
		    }
		});

	}

	public void close() {
		try {
			Thread.sleep(100);
		} catch (InterruptedException e) {
		} // wait a few second in case threads need to shut down nicely
		try {
			udpr.stopRunning();
		} catch (Exception e) {
			// TODO: handle exception
			System.out.println("UDP Logger failed with error " + e);
		}
		System.exit(0);
	}

	
	public static void main(String[] args) {
		new LTMainMulti(550,750);
	}

}
