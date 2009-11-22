package net.electroland.installutils.mgmt;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.WindowEvent;
import java.io.BufferedReader;
import java.io.IOException;
import java.net.UnknownHostException;

import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.KeyStroke;

import net.miginfocom.swing.MigLayout;

public class ClientJFrame extends JFrame implements ActionListener, Runnable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1163302714818806541L;
	private JMenuBar menuBar;
	private JMenu menu;
	private JMenuItem start, stop, clear, reconnect;
	private JTextArea output;
	private Connection c;
	private BufferedReader responseStream;

	public ClientJFrame(Connection c, int x, int y)
	{
		super(c.address + ':' + c.port);
		
		this.c = c;

		this.setLayout(new MigLayout());
		this.setLocation(x,y);
		this.setSize(550, 450);
		
		//Create the menu bar.
		menuBar = new JMenuBar();
		

		//Build the first menu.
		menu = new JMenu("Options");
		menu.setMnemonic(KeyEvent.VK_A);
		menu.getAccessibleContext().setAccessibleDescription(
		        "The only menu in this program that has menu items");
		menuBar.add(menu);

		// start option
		start = new JMenuItem("Start",
		                         KeyEvent.VK_R);
		start.setAccelerator(KeyStroke.getKeyStroke(
		        KeyEvent.VK_R, ActionEvent.CTRL_MASK));
		start.getAccessibleContext().setAccessibleDescription(
		        "(re)start this client");
		start.addActionListener(this);
		menu.add(start);

		// stop option
		stop = new JMenuItem("Stop",
		                         KeyEvent.VK_S);
		stop.setAccelerator(KeyStroke.getKeyStroke(
		        KeyEvent.VK_S, ActionEvent.CTRL_MASK));
		stop.getAccessibleContext().setAccessibleDescription(
		        "stop this client");
		stop.addActionListener(this);
		menu.add(stop);


		// clear option
		clear = new JMenuItem("Clear");
		clear.getAccessibleContext().setAccessibleDescription(
		        "clear the output window");
		clear.addActionListener(this);
		menu.add(clear);
		

		// clear option
		reconnect = new JMenuItem("Reset connection");
		reconnect.getAccessibleContext().setAccessibleDescription(
		        "reconnect to the client");
		reconnect.addActionListener(this);
		menu.add(reconnect);
		
		this.add(menuBar, "span 1, wrap");

		// text area
		output = new JTextArea(24, 80);
		output.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(output);

        this.add(scrollPane, "span1, grow");
		this.setVisible(true);
		this.addWindowListener(new java.awt.event.WindowAdapter()
		{
			public void windowClosing(WindowEvent winEvt)
			{
				System.exit(0);
			}
		});
	}

	public void start()
	{
		try {
			c.start();
		} catch (UnknownHostException e1) {
			output.append(e1.toString());
		} catch (IOException e1) {
			output.append(e1.toString());
		}finally{
			output.append("\n\r");
		}		
	}
	
	public void stop()
	{
		try {
			c.stop();
		} catch (UnknownHostException e1) {
			output.append(e1.toString());
		} catch (IOException e1) {
			output.append(e1.toString());
		}finally{
			output.append("\n\r");
		}		
	}

	public void close()
	{
		try {
			output.append("connection reset.");
			c.close();
		} catch (UnknownHostException e1) {
			output.append(e1.toString());
		} catch (IOException e1) {
			output.append(e1.toString());
		}finally{
			output.append("\n\r");
		}		
	}

	public void actionPerformed(ActionEvent e) {
		if ("Stop".equalsIgnoreCase(e.getActionCommand())){
			stop();
		}else if ("Start".equalsIgnoreCase(e.getActionCommand())){
			start();
		}else if ("Clear".equalsIgnoreCase(e.getActionCommand())){
			output.setText("");
		}else if ("Reset connection".equalsIgnoreCase(e.getActionCommand())){
			close();
		}
	}

	public void run() {
		while (true)
		{
			// kludgy.  get the responsStream directly from the Connection object over and
			// over again.  because the Connection object isn't updating us when it
			// connected.
			responseStream = c.responseStream;
			try {
				if (responseStream != null && responseStream.ready())
				{
					output.append(responseStream.readLine());
					output.append("\n\r");
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}