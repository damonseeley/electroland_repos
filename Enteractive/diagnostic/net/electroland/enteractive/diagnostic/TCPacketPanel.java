package net.electroland.enteractive.diagnostic;

import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSeparator;
import javax.swing.JTextField;

import net.electroland.enteractive.utils.HexUtils;
import net.miginfocom.swing.MigLayout;

public class TCPacketPanel extends JPanel {

	private int w;
	private int h;
	private TCBroadcaster tcb;
	
	private int tfWidth = 59;

	public JTextField bytes01;
	public JTextField bytes02;
	public JTextField bytes03;
	public JTextField bytes04;
	public JTextField bytes05;
	public JTextField bytes06;
	
	private JButton send01;
	private JButton send02;
	private JButton send03;
	private JButton send04;
	private JButton send05;
	private JButton send06;

	public TCPacketPanel(int inset, TCBroadcaster tcb) {

		this.tcb = tcb;

		String insetStr = "inset " + inset;
		JPanel p = new JPanel(new MigLayout(insetStr,""));
		
//		p.add(new JLabel("Packets"), "split, span");
//		p.add(new JSeparator(), "growx, wrap");

		Font packetFont = new Font("Consolas", Font.PLAIN, 10);
		
		send01 = new JButton("Send Packet 1");
		send01.setOpaque(true);
		send01.addActionListener(new sendPacket());
		p.add(send01,"left");
		bytes01 = new JTextField(TCDiagnosticMain.properties.get("bytes01"),tfWidth);
		bytes01.setFont(packetFont);
		p.add(bytes01, "span, growx");
		
		send02 = new JButton("Send Packet 2");
		send02.setOpaque(true);
		send02.addActionListener(new sendPacket());
		p.add(send02,"left");
		bytes02 = new JTextField(TCDiagnosticMain.properties.get("bytes02"),tfWidth);
		bytes02.setFont(packetFont);
		p.add(bytes02, "span, growx");
		
		send03 = new JButton("Send Packet 3");
		send03.setOpaque(true);
		send03.addActionListener(new sendPacket());
		p.add(send03,"left");
		bytes03 = new JTextField(TCDiagnosticMain.properties.get("bytes03"),tfWidth);
		bytes03.setFont(packetFont);
		p.add(bytes03, "span, growx");
		
		send04 = new JButton("Send Packet 4");
		send04.setOpaque(true);
		send04.addActionListener(new sendPacket());
		p.add(send04,"left");
		bytes04 = new JTextField(TCDiagnosticMain.properties.get("bytes04"),tfWidth);
		bytes04.setFont(packetFont);
		p.add(bytes04, "span, growx");
		
		send05 = new JButton("Send Packet 5");
		send05.setOpaque(true);
		send05.addActionListener(new sendPacket());
		p.add(send05,"left");
		bytes05 = new JTextField(TCDiagnosticMain.properties.get("bytes05"),tfWidth);
		bytes05.setFont(packetFont);
		p.add(bytes05, "span, growx");
		
		send06 = new JButton("Send Packet 6");
		send06.setOpaque(true);
		send06.addActionListener(new sendPacket());
		p.add(send06,"left");
		bytes06 = new JTextField(TCDiagnosticMain.properties.get("bytes06"),tfWidth);
		bytes06.setFont(packetFont);
		p.add(bytes06, "span, growx");
		
		add(p);

	}
	
	public class sendPacket implements ActionListener {
        public void actionPerformed(ActionEvent e) {
        	//tcb.sendOne("DUMMY STRING FOR NOW");
        	
        	//System.out.println(e.getActionCommand());
        	
        	if (e.getActionCommand() == "Send Packet 1"){
        		tcb.sendOne(bytes01.getText());
        	}
        	
        	if (e.getActionCommand() == "Send Packet 2"){
        		tcb.sendOne(bytes02.getText());
        	}
        	
        	if (e.getActionCommand() == "Send Packet 3"){
        		tcb.sendOne(bytes03.getText());
        	}
        	
        	if (e.getActionCommand() == "Send Packet 4"){
        		tcb.sendOne(bytes04.getText());
        	}
        	
        	if (e.getActionCommand() == "Send Packet 5"){
        		tcb.sendOne(bytes05.getText());
        	}
        	
        	if (e.getActionCommand() == "Send Packet 6"){
        		tcb.sendOne(bytes06.getText());
        	}
        }
    }

}
