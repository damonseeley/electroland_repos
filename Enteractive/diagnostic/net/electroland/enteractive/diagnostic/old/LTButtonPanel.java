package net.electroland.enteractive.diagnostic.old;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.AbstractButton;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSeparator;

import net.miginfocom.swing.MigLayout;

public class LTButtonPanel extends JPanel {

	private int w;
	private int h;
	private LTUDPBroadcasterManager lM;
	
	private int tfWidth = 30;

	private JButton sendOne;
	private JButton sendStream;


	public LTButtonPanel(int inset, LTUDPBroadcasterManager lM) {

		this.w = w;
		this.h = h;
		this.lM = lM;

		String insetStr = "inset " + inset;
		MigLayout bl = new MigLayout(insetStr,"[left]20[left]20[left]","[][]");
		JPanel p = new JPanel(bl);
		
		sendOne = new JButton("Send 1 Packet");
		sendOne.setOpaque(true);
		sendOne.addActionListener(new sendOneAction());
		//sendOne.setActionCommand("disable");
		sendStream = new JButton("Start Stream");
		sendStream.setOpaque(true);
		sendStream.addActionListener(new toggleStreamAction());
		
		p.add(sendOne,"left");
		p.add(sendStream,"trailing");
		
		add(p);
	}
	
	public class sendOneAction implements ActionListener {
        public void actionPerformed(ActionEvent e) {
        	//System.out.println("change");
        	lM.sendOne();
        }
    }
	
	public class toggleStreamAction implements ActionListener {
        public void actionPerformed(ActionEvent e) {
        	//System.out.println("change");
        	if (sendStream.getText() == "Start Stream") {
        		sendStream.setText("Stop Stream");
        		lM.startBroadcast();
        	} else {
        		sendStream.setText("Start Stream");
        		lM.stopBroadcast();
        	}
        }
    }
	
	
	
	
}
