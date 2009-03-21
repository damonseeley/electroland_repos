package net.electroland.elvis.manager;

import java.awt.Dimension;
import java.awt.event.WindowEvent;

import javax.swing.JFrame;

public class MainFrame extends JFrame {
       
//	CreatorToolBar ctb;
	ImagePanel imagePanel; ;
	SettingsTab settings;
	
	public static MainFrame THE_FRAME;
	
	

	
	public MainFrame(int w, int h) {
		super("ElVis");
		if(THE_FRAME != null) return;
		THE_FRAME = this;
		
		imagePanel = new ImagePanel(w,h);
		settings = new SettingsTab();
		
		setLayout(null);
		
//		ctb = new CreatorToolBar(imagePanel);
//		add(ctb);
//		Dimension buttonSize = ctb.getPreferredSize();
//		ctb.setBounds(getInsets().left, getInsets().top, (int) buttonSize.getWidth(), (int) buttonSize.getHeight());
		

		add(imagePanel);
		Dimension imageSize = imagePanel.getPreferredSize();

		int minHeight = 500;
		int frameHeight = (imageSize.getHeight()> minHeight)?(int) imageSize.getHeight():minHeight;
		
		imagePanel.setBounds( getInsets().left, getInsets().top, (int) imageSize.getWidth(),frameHeight);
		addMouseListener(imagePanel);
		addMouseMotionListener(imagePanel);
		addKeyListener(imagePanel);
		
		add(settings);
		Dimension settinsSize = settings.getPreferredSize();
		settings.setBounds((int) ( imagePanel.getWidth())+getInsets().left, getInsets().top, (int) settinsSize.getWidth(),frameHeight);
		
		setVisible(true);
		setSize((int)(  imagePanel.getWidth()+settinsSize.getWidth())+getInsets().left+getInsets().right, (int) frameHeight+getInsets().top + getInsets().bottom);
		imagePanel.xOffset = getInsets().left;
		imagePanel.yOffset = getInsets().top;
		setResizable(true);
		imagePanel.requestFocus();	
		
		addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(WindowEvent winEvt) {
		    	close();
		    }
		});
		
	}
	
	public void close() {
		imagePanel.stop();
		try {
			Thread.sleep(100);
		} catch (InterruptedException e) {
		} // wait a few second in case threads need to shut down nicely
		System.exit(0);
	}

	public static void main(String[] args) {
		new MainFrame(240,180);
	}

}
