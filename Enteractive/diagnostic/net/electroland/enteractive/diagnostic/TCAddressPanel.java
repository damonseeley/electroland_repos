package net.electroland.enteractive.diagnostic;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSeparator;
import javax.swing.JSlider;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.electroland.enteractive.diagnostic.TCPacketPanel.sendPacket;
import net.electroland.enteractive.diagnostic.old.LTUDPBroadcasterManager;
import net.electroland.enteractive.utils.HexUtils;
import net.miginfocom.swing.MigLayout;

public class TCAddressPanel extends JPanel {

	private int w;
	private int h;
	private TCBroadcaster tcb;
	
	private int tfWidth = 30;

	private JTextField ipAddress;
	private JTextField udpPort;
	private JButton setAddress;


	public TCAddressPanel(int inset, String ip, Integer intPort, TCBroadcaster tcb) {

		this.w = w;
		this.h = h;
		this.tcb = tcb;

		String insetStr = "inset " + inset;
		JPanel p = new JPanel(new MigLayout(insetStr,""));
		//JPanel p = new JPanel(new MigLayout(insetStr,"20[left]20[left]20"));

		p.add(new JLabel("IP Address"), "gap 10");
		ipAddress = new JTextField(ip,tfWidth);
		ipAddress.addKeyListener(new addressKeyAction());
		p.add(ipAddress, "span, grow");

		p.add(new JLabel("UDP Port"), "gap 10");
		udpPort = new JTextField(intPort.toString(),tfWidth);
		udpPort.addKeyListener(new addressKeyAction());
		p.add(udpPort, "span, grow");
		
		setAddress = new JButton("Set Socket Properties");
		setAddress.setOpaque(true);
		setAddress.addActionListener(new setAddressAction());
		p.add(setAddress,"span, center");

		add(p);


	}
	
	// reset the address and port in tcb and reopen socket
	public class setAddressAction implements ActionListener {
        public void actionPerformed(ActionEvent e) {
        	tcb.setNewSocketParameters(ipAddress.getText(),Integer.parseInt(udpPort.getText()));
        }
    }

	
	// removed due to multiple broadcasters
/*	public class ipAddressAction implements ActionListener {
        public void actionPerformed(ActionEvent e) {
        	//System.out.println("change");
        	tcb.setIpAddress(ipAddress.getText());
        }
    }
*/	
	
	public class addressKeyAction implements KeyListener {
        public void keyReleased(KeyEvent ke) {
        	//System.out.println(ke.getKeyCode());
        	// set the address on a return
        	if (ke.getKeyCode() == 10){
        		setAddress.doClick();
        	}
        }
        
        public void keyPressed (KeyEvent ke) {
        	//System.out.println(ke.getKeyCode());
        }
        
        public void keyTyped (KeyEvent ke) {
        	//System.out.println(ke.getKeyCode());
        }
    }

	
	private void updateAddressString() {
//		completePacket.setText("FF" + cmdByte.getText() + dataBytes.getText() + "FE");
//		//completePacket.setText(startByte.getText() + addressByte.getText() + cmdByte.getText() + dataBytes.getText());
//		tcb.setPacketString(completePacket.getText());
	}
}
