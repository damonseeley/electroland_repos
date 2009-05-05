package net.electroland.broadcast.example;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import net.electroland.broadcast.server.FlexBroadcasterUtil;
import net.electroland.broadcast.server.XMLSocketBroadcaster;

public class FlexBroadcaster extends JFrame implements ActionListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private XMLSocketBroadcaster xmlsb;
	JTextArea message;

	public FlexBroadcaster(int port){
		super("Flex Broadcaster");
		this.setSize(800, 800);
		this.setLayout(new BorderLayout());
		message = new JTextArea(30, 40);
		message.setText("<pool><fish x=\"1\" y=\"1\" v=\"5\" " +
									"d=\"10\" o=\"15\" id=\"" + 
									FlexBroadcasterUtil.getUniqueId() + 
									"\" " +
									"t=\"0\" s=\"0\" p=\"1.0\" f=\"45\"/></pool>");
		JScrollPane scrollingArea = new JScrollPane(message);
		this.add(scrollingArea, BorderLayout.CENTER);

		JButton sendButton = new JButton();
		sendButton.setText("Send");
		sendButton.addActionListener(this);
		this.add(scrollingArea, BorderLayout.SOUTH);
	    this.add(sendButton);
	    this.pack();

        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setVisible(true);
        
        xmlsb = new XMLSocketBroadcaster(port);
		xmlsb.start();
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		if (args == null || args.length != 1){
			System.out.println("Usage: java FlexBroadcasterTest [port]");
		}else{
			try{
				int port = Integer.parseInt(args[0]);
				new FlexBroadcaster(port);
				// add some threaded delayed calls to send(XMLSocketMessage)
				// here.  better yet, a window with a big text field for
				// sending xml to.
				
			}catch (NumberFormatException e){
				System.out.println("Invalid port: " + args[0]);
			}
		}
	}

	public void actionPerformed(ActionEvent arg0) {
		xmlsb.send(new TestXMLSocketMessage(message.getText()));
	}
}