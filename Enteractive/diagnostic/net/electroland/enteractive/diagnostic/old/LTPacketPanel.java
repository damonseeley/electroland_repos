package net.electroland.enteractive.diagnostic.old;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSeparator;
import javax.swing.JSlider;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.electroland.enteractive.utils.HexUtils;
import net.miginfocom.swing.MigLayout;

public class LTPacketPanel extends JPanel {

	private int w;
	private int h;
	private LTUDPBroadcasterManager lM;
	
	private int tfWidth = 30;

	private JSlider fpsSlider;
	private JSlider cmdSlider;
	private JSlider dataSlider;
	private JLabel frameRate;
	private JTextField ipAddress;
	private JTextField startByte;
	private JTextField addressByte;
	private JTextField cmdByte;
	private JTextField dataBytes;
	private JTextField completePacket;

	public LTPacketPanel(int inset, String ip, LTUDPBroadcasterManager lM) {

		this.w = w;
		this.h = h;
		this.lM = lM;

		String insetStr = "inset " + inset;
		JPanel p = new JPanel(new MigLayout(insetStr,""));
		
		p.add(new JLabel("Packet Components"), "split, span");
		p.add(new JSeparator(), "growx, wrap");

		p.add(new JLabel("IP Address Start"), "gap 10");
		ipAddress = new JTextField(ip);
		//ipAddress.addActionListener(new ipAddressAction());
		p.add(ipAddress, "span, grow");
		//ipAddress.addKeyListener(new packetKeyAction());
		
		/*p.add(new JLabel("Address Byte"), "gap 10");
		addressByte = new JTextField("00");
		addressByte.addKeyListener(new packetKeyAction());
		p.add(addressByte, "span, growx");
		
		p.add(new JLabel("Address Selector"), "gap 10, split, span");
		addressSlider = new JSlider(0, 255, 0);
		p.add(addressSlider, "span, growx, wrap");
		addressSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				//System.out.println(HexUtils.decimalToHex(addressSlider.getValue()));
				addressByte.setText(HexUtils.decimalToHex(addressSlider.getValue()));
				updatePacketString();	
			}
		});*/
		
		p.add(new JLabel("Cmd Byte"), "gap 10");
		cmdByte = new JTextField("01");
		cmdByte.addKeyListener(new packetKeyAction());
		p.add(cmdByte, "span, growx");
		
		p.add(new JLabel("Cmd Selector"), "gap 10, split, span");
		cmdSlider = new JSlider(0, 253, 0);
		p.add(cmdSlider, "span, growx, wrap");
		cmdSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				//System.out.println(HexUtils.decimalToHex(addressSlider.getValue()));
				cmdByte.setText(HexUtils.decimalToHex(cmdSlider.getValue()));
				updatePacketString();	
			}
		});
		
		
		p.add(new JLabel("Data Bytes"), "gap 10");
		dataBytes = new JTextField("0203040506070809");
		dataBytes.addKeyListener(new packetKeyAction());
		p.add(dataBytes, "span, growx");
		
		p.add(new JLabel("Data Selector"), "gap 10, split, span");
		dataSlider = new JSlider(0, 253, 0);
		p.add(dataSlider, "span, growx, wrap");
		dataSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				//System.out.println(HexUtils.decimalToHex(addressSlider.getValue()));
				String dataString = "";
				for (int t=0;t<8;t++){
					dataString += HexUtils.decimalToHex(dataSlider.getValue());
				}
				dataBytes.setText(dataString);
				updatePacketString();	
			}
		});
		
		

		p.add(new JLabel("Complete Packet"), "split, span, gaptop 10");
		p.add(new JSeparator(), "growx, wrap, gaptop 10");
		completePacket = new JTextField(tfWidth);
		p.add(completePacket, "span, growx, wrap, gap 10");

		
		p.add(new JLabel("Framerate"), "split, span, gaptop 20");
		fpsSlider = new JSlider(1, 120, 30);
		p.add(fpsSlider, "span, growx, gaptop 20");
		fpsSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				//System.out.println(fpsSlider.getValue());
				updateFramerate();
				frameRate.setText(fpsSlider.getValue() + " fps");
			}
		});
		frameRate = new JLabel("30 fps");
		p.add(frameRate, "span, wrap, gaptop 20");

		add(p);
		
		updateFramerate();
		updatePacketString();

	}

	
	// removed due to multiple broadcasters
/*	public class ipAddressAction implements ActionListener {
        public void actionPerformed(ActionEvent e) {
        	//System.out.println("change");
        	lM.setIpAddress(ipAddress.getText());
        }
    }
*/	
	
	public class packetKeyAction implements KeyListener {
        public void keyReleased(KeyEvent ke) {
        	updatePacketString();
        }
        
        public void keyPressed (KeyEvent ke) {
        	updatePacketString();
        }
        
        public void keyTyped (KeyEvent ke) {
        	updatePacketString();
        }
    }
	
	private void updateFramerate(){
		lM.setFramerate(fpsSlider.getValue());
	}
	
	private void updatePacketString() {
		completePacket.setText("FF" + cmdByte.getText() + dataBytes.getText() + "FE");
		//completePacket.setText(startByte.getText() + addressByte.getText() + cmdByte.getText() + dataBytes.getText());
		lM.setPacketString(completePacket.getText());
	}
}
