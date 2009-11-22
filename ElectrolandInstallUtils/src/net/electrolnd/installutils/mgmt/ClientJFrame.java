package net.electrolnd.installutils.mgmt;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
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

public class ClientJFrame extends JFrame implements ActionListener{

	private JMenuBar menuBar;
	private JMenu menu;
	private JMenuItem start, stop;
	private JTextArea output;
	private Connection c;

	public ClientJFrame(Connection c)
	{
		super(c.address + ':' + c.port);
		
		this.c = c;

		this.setLayout(new MigLayout());

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

		this.add(menuBar, "span 1, wrap");

		// text area
		output = new JTextArea(24, 80);
		output.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(output);

        this.add(scrollPane, "span1, grow");
		this.setSize(550, 450);
		this.setVisible(true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if ("Stop".equalsIgnoreCase(e.getActionCommand())){
			try {
				output.append(c.stop());
			} catch (UnknownHostException e1) {
				output.append(e1.toString());
			} catch (IOException e1) {
				output.append(e1.toString());
			}finally{
				output.append("\n\r");
			}
		}else if ("Start".equalsIgnoreCase(e.getActionCommand())){
			try {
				output.append(c.start());
			} catch (UnknownHostException e1) {
				output.append(e1.toString());
			} catch (IOException e1) {
				output.append(e1.toString());
			}finally{
				output.append("\n\r");
			}
		}
	}
}