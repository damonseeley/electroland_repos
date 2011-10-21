package net.electroland.modbus.core;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.color.ColorSpace;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;

import javax.swing.BorderFactory;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSeparator;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;
import net.wimpi.modbus.util.BitVector;

import org.apache.log4j.Logger;

public class EIAInstallTool extends JFrame implements Runnable  {

	protected JButton startButton, stopButton;
	protected JTextField ipAddressInput;
	protected ButtonGroup buttonGroup;
	protected JLabel ipCmds, sensorOutput;

	SensorPanel sp;

	private MTMThread mtmt;
	private String startIP;
	private int startFramerate;

	private int windowWidth,windowHeight;
	private int sensorWidth,sensorHeight;

	static Logger logger = Logger.getLogger(EIAInstallTool.class);

	public EIAInstallTool() {

		super("EIA Setup Tool");

		windowWidth = 1024;
		windowHeight = 768;
		sensorWidth = 992;
		sensorHeight = 600;

		startIP = "192.168.247.61";
		startFramerate = 60;

		JPanel ipPanel = new JPanel();
		ipPanel.setBorder(BorderFactory.createTitledBorder("IP Address Info"));
		this.setLayout(new MigLayout(""));
		this.add(ipPanel, "center, growx, wrap");

		ipPanel.setLayout(new MigLayout(""));

		ipAddressInput = new JTextField();
		ipAddressInput.setText(startIP);
		ipPanel.add(ipAddressInput, "span ,growx, wrap");
		ipPanel.add(new JSeparator(), "span, growx, wrap");
		startButton = new JButton("Start");
		ipPanel.add(startButton, "center");
		stopButton = new JButton("Stop");
		ipPanel.add(stopButton, "center");

		startButton.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e) {
				resetMTMT(ipAddressInput.getText());
			}
		});

		stopButton.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e) {
				//logger.info(e);
				killMTMT();
			}
		});

		sp = new SensorPanel(sensorWidth,sensorHeight);
		this.add(sp);

		//setup window
		this.setVisible(true);
		this.setSize(windowWidth, windowHeight);

		this.addWindowListener(
				new java.awt.event.WindowAdapter() {
					public void windowClosing(WindowEvent winEvt) {
						//fix
						close();
					}
				});

		mtmt = new MTMThread(startIP,startFramerate,sp);
		
		Thread t=new Thread (this);
		t. start();

	}

	public void run() {
		while(true) {
			//logger.info("I'm running");
		}
	}

	private void resetMTMT(String newip){
		logger.info("Resetting MTMT");
		mtmt.connectMTM(newip);
	}

	private void killMTMT() {
		logger.info("Stopping MTMT");
		mtmt.disconnectMTM();
	}




	public static void printOutput(byte[] bytes, String label)
	{
		BitVector bv = BitVector.createBitVector(bytes);

		for (int i=0; i < bv.size(); i++)
		{
			System.out.print(bv.getBit(i) ? '1' : '0');
			if ((i+1) % 8 == 0){
				System.out.print(' ');
			}
		}
		System.out.println("<-- us on " + label);

	}


	private void close(){
		mtmt.stopClean();
		System.exit(0);
	}


	public static void main(String[] args) {
		// TODO Auto-generated method stub
		EIAInstallTool eiait = new EIAInstallTool();

	}

}
