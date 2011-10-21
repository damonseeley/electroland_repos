package net.electroland.EIA.core;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.color.ColorSpace;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;

import javax.swing.BorderFactory;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSeparator;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;
import org.apache.log4j.Logger;

public class EIAFrame extends JFrame  {

	protected JButton startButton, stopButton;
	protected JTextField ipAddressInput;
	protected ButtonGroup buttonGroup;
	protected JLabel ipCmds, sensorOutput;

	EIAPanel sp;

	private int windowWidth,windowHeight;
	private int panelWidth,panelHeight;

	static Logger logger = Logger.getLogger(EIAFrame.class);

	public EIAFrame(int width, int height) {

		super("EIA Setup Tool");

		windowWidth = 1024;
		windowHeight = 768;
		panelWidth = 992;
		panelHeight = 640;


		JPanel controlPanel = new JPanel();
		controlPanel.setBorder(BorderFactory.createTitledBorder("EIA Controls"));
		this.setLayout(new MigLayout(""));
		this.add(controlPanel, "center, growx, wrap");

		controlPanel.setLayout(new MigLayout(""));


		ipAddressInput = new JTextField();
		ipAddressInput.setText(startIP);
		controlPanel.add(ipAddressInput, "span ,growx, wrap");
		controlPanel.add(new JSeparator(), "span, growx, wrap");
		startButton = new JButton("Start");
		controlPanel.add(startButton, "center");
		stopButton = new JButton("Stop");
		controlPanel.add(stopButton, "center");

		startButton.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e) {
			}
		});

		stopButton.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e) {
				//logger.info(e);
			}
		});

		sp = new EIAPanel(panelWidth,panelHeigh);
		this.add(sp);


		//setup window
		this.setVisible(true);
		this.setSize(windowWidth, windowHeight);

		this.addWindowListener(
				new java.awt.event.WindowAdapter() {
					public void windowClosing(WindowEvent winEvt) {
						//fix
						close();
					}
				});




	}

	public BufferedImage getPanelImage() {
		BufferedImage bi = new BufferedImage(16, 16, 16);
		return bi;
	}




	private void close(){
		System.exit(0);
	}



}
