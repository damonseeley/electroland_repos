package net.electroland.elvis.manager;

import java.awt.Dimension;
import java.awt.event.WindowEvent;
import java.net.SocketException;
import java.net.UnknownHostException;

import javax.swing.JFrame;

import net.electroland.elvis.util.ElProps;

public class MainFrame extends JFrame {
       
//	CreatorToolBar ctb;
	ImagePanel imagePanel; ;
	SettingsTab settings;
	
	public static MainFrame THE_FRAME;
	
	

	
	public MainFrame(ElProps props, int w, int h) throws SocketException, UnknownHostException {
		super("ElVis");
		if(THE_FRAME != null) return;
		THE_FRAME = this;
			
		imagePanel = new ImagePanel(props, w,h);
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

	public static void main(String[] args) throws SocketException, UnknownHostException {
		ElProps p;
		if(args.length > 0) {
			p= ElProps.init(args[0]);
		} else {
			p = ElProps.init("blobTracker.props");
		}
		new MainFrame(p, 640,480);
	}

}
