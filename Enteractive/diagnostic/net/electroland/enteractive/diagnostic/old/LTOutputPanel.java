package net.electroland.enteractive.diagnostic.old;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSeparator;
import javax.swing.JSlider;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.electroland.enteractive.udpUtils.UDPReceiver;
import net.miginfocom.swing.MigLayout;

public class LTOutputPanel extends JPanel {

	private int w;
	private int h;
	
	private int tfWidth = 30;

	private JTextArea returnTA;
	private JScrollPane jScrollPane1;
	
	public UDPReceiver udpr;

	public LTOutputPanel(int inset) {

		this.w = w;
		this.h = h;

		String insetStr = "inset " + inset;
		JPanel p = new JPanel(new MigLayout(insetStr,""));
		
		returnTA = new JTextArea();
		returnTA.setColumns(40);
		returnTA.setLineWrap(true);
		returnTA.setRows(5);
		returnTA.setWrapStyleWord(true);
		returnTA.setEditable(true);
		jScrollPane1 = new JScrollPane(returnTA);

		p.add(jScrollPane1);
		
		add(p);

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
