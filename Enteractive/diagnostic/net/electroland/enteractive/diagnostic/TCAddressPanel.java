package net.electroland.enteractive.diagnostic;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.net.SocketException;
import java.net.UnknownHostException;

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
	private JTextField sendPort;
	private JTextField rcvPort;
	private JButton setAddress;


	public TCAddressPanel(int inset, String ip, Integer intSendPort, Integer intRcvPort, TCBroadcaster tcb) {

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

		p.add(new JLabel("Send Port"), "gap 10");
		sendPort = new JTextField(intSendPort.toString(),tfWidth);
		sendPort.addKeyListener(new addressKeyAction());
		p.add(sendPort, "span, grow");
		
		p.add(new JLabel("Receive Port"), "gap 10");
		rcvPort = new JTextField(intRcvPort.toString(),tfWidth);
		rcvPort.addKeyListener(new addressKeyAction());
		p.add(rcvPort, "span, grow");
		
		setAddress = new JButton("Set Socket Properties");
		setAddress.setOpaque(true);
		setAddress.addActionListener(new setAddressAction());
		p.add(setAddress,"span, center");

		add(p);


	}
	
	// reset the address and port in tcb and reopen socket
	public class setAddressAction implements ActionListener {
        public void actionPerformed(ActionEvent e) {
        	tcb.setNewSocketParameters(ipAddress.getText(),Integer.parseInt(sendPort.getText()));
        	
        	// update the props file
        	TCDiagnosticMain.properties.put("IPAddress", ipAddress.getText());
        	TCDiagnosticMain.properties.put("sendPort", sendPort.getText());
        	TCDiagnosticMain.properties.put("rcvPort", rcvPort.getText());
        	
        	try {
				TCDiagnosticMain.udpr.setNewRcvPort(Integer.parseInt(rcvPort.getText()));
			} catch (NumberFormatException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			} catch (SocketException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			} catch (UnknownHostException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
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
