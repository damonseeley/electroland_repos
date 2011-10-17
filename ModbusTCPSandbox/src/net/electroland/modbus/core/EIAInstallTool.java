package net.electroland.modbus.core;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.io.BufferedReader;
import java.io.InputStreamReader;

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

import net.miginfocom.swing.MigLayout;
import net.wimpi.modbus.util.BitVector;

import org.apache.log4j.Logger;

public class EIAInstallTool extends JFrame{

protected JButton startButton, stopButton;
protected JTextField ipAddressInput;
protected ButtonGroup buttonGroup;
protected JLabel ipCmds, sensorOutput;

private MTMThread mtmt;
private String startIP;
private int startFramerate;
	
	static Logger logger = Logger.getLogger(EIAInstallTool.class);

	public EIAInstallTool(int width, int height) {
		
		super("EIA Setup Tool");
		
		startIP = "192.168.247.22";
		startFramerate = 60;
		
		JPanel ipPanel = new JPanel();
		ipPanel.setBorder(BorderFactory.createTitledBorder("IP Address Info"));
		this.setLayout(new MigLayout(""));
		this.add(ipPanel, "center, growx, wrap");
		
		ipPanel.setLayout(new MigLayout(""));

		
		ipAddressInput = new JTextField();
		ipPanel.add(ipAddressInput, "span2 ,growx, wrap");
		ipPanel.add(new JSeparator(), "span, growx, wrap");
		startButton = new JButton("Start");
		ipPanel.add(startButton, "center");
		stopButton = new JButton("Stop");
		ipPanel.add(stopButton, "center");
		
		startButton.addActionListener(new ActionListener(){
	        public void actionPerformed(ActionEvent e) {
	        	resetMTMT("0.0.0.0");
	        }
		});
		
		stopButton.addActionListener(new ActionListener(){
	        public void actionPerformed(ActionEvent e) {
	        	//logger.info(e);
	        	killMTMT();
	        }
		});
		
		//setup window
		this.setVisible(true);
		this.setSize(width, height);

		this.addWindowListener(
			new java.awt.event.WindowAdapter() {
			    public void windowClosing(WindowEvent winEvt) {
			    	//fix
			    	//close();
			    }
		});
		
		mtmt = new MTMThread(startIP,startFramerate);
		
		
		

	}
	
	private void resetMTMT(String newip){
		logger.info("Resetting MTMT");
		mtmt.stopClean();
		//mtmt = new MTMThread(newip,startFramerate);
	}
	
	private void killMTMT() {
		logger.info("Killing MTMT");
		mtmt.stopClean();
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
		EIAInstallTool eiait = new EIAInstallTool(1024, 768);

	}

}
