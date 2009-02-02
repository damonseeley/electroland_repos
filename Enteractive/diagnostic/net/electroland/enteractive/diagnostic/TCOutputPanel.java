package net.electroland.enteractive.diagnostic;

import java.awt.Dimension;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.net.SocketException;
import java.net.UnknownHostException;

import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSeparator;
import javax.swing.JSlider;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.electroland.enteractive.diagnostic.TCAddressPanel.setAddressAction;
import net.electroland.enteractive.udpUtils.UDPReceiver;
import net.miginfocom.swing.MigLayout;

public class TCOutputPanel extends JPanel {

	private int w;
	private int h;
	
	private int tfWidth = 63;

	private JTextArea returnTA;
	private JScrollPane jScrollPane1;
	private JButton clear;
	
	public UDPReceiver udpr;

	public TCOutputPanel(int inset) {

		this.w = w;
		this.h = h;

		String insetStr = "inset " + inset;
		JPanel p = new JPanel(new MigLayout(insetStr,""));
		
		Font returnFont = new Font("Consolas", Font.PLAIN, 11);
		
		returnTA = new JTextArea();
		returnTA.setColumns(tfWidth);
		returnTA.setFont(returnFont);
		returnTA.setLineWrap(true);
		returnTA.setRows(5);
		returnTA.setWrapStyleWord(true);
		returnTA.setEditable(true);
		jScrollPane1 = new JScrollPane(returnTA);
		p.add(jScrollPane1);
		
		clear = new JButton("Clear");
		clear.setOpaque(true);
		clear.addActionListener(new clearAction());
		p.add(clear,"span, center");
		
		add(p);

	}
	
	public class clearAction implements ActionListener {
        public void actionPerformed(ActionEvent e) {
        	returnTA.setText("");
        }
    }
	
	
	
	public JTextArea getOutputField() {
		return returnTA;
	}
	
	public class packetAction implements ActionListener {
        public void actionPerformed(ActionEvent e) {
        	//System.out.println("change");
        }
    }
	
	public class packetKeyAction implements KeyListener {
        public void keyReleased(KeyEvent ke) {
        	update();
        }
        
        public void keyPressed (KeyEvent ke) {
        	update();
        }
        
        public void keyTyped (KeyEvent ke) {
        	update();
        }
        
        private void update() {
        	//System.out.println("change");
        }
    }
}
