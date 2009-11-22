package net.electroland.installutils.mgmt;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.io.IOException;
import java.net.UnknownHostException;
import java.util.Collection;
import java.util.Iterator;

import javax.swing.JButton;
import javax.swing.JFrame;

import net.miginfocom.swing.MigLayout;

public class MasterJFrame extends JFrame implements ActionListener{

	/**
	 * 
	 */
	private static final long serialVersionUID = -4463633428581584781L;
	private JButton stopAll, startAll;
	private Collection<ClientJFrame> clients;
	
	public MasterJFrame(Collection<ClientJFrame> clients)
	{
		super("Master Director");
		
		this.clients = clients;
		
		this.setLayout(new MigLayout());
		this.setSize(250, 100);
		
		stopAll = new JButton("stop all");
		stopAll.addActionListener(this);
		this.add(stopAll, "span 1");

		startAll = new JButton("start all");
		startAll.addActionListener(this);
		this.add(startAll, "span 1, wrap");
		
		this.setVisible(true);

		this.addWindowListener(new java.awt.event.WindowAdapter()
		{
			public void windowClosing(WindowEvent winEvt)
			{
				System.exit(0);
			}
		});	
	}

	public void actionPerformed(ActionEvent e) {
		if ("stop all".equalsIgnoreCase(e.getActionCommand())){	
			Iterator<ClientJFrame> i = clients.iterator();
			while (i.hasNext()){
				i.next().stop();
			}
		}else if ("start all".equalsIgnoreCase(e.getActionCommand())){
			Iterator<ClientJFrame> i = clients.iterator();
			while (i.hasNext()){
				i.next().start();
			}			
		}
	}
}
