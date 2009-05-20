package net.electroland.elvisVideoProcessor.demo;

import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;

import javax.swing.JFrame;

import net.electroland.elvisVideoProcessor.ElProps;
import net.electroland.elvisVideoProcessor.LAFaceVideoProcessor;
import net.electroland.elvisVideoProcessor.ui.LAFaceFrame;

public class Demo extends JFrame {
	LAFaceVideoProcessor vidProcessor;
	BufferedImage[] imgs;
	DemoThread demoThread;


	public Demo(String name, LAFaceVideoProcessor vidProcessor) {
		super(name);
		this.vidProcessor = vidProcessor;
		setSize(680, 520);
		setVisible(true);
		setResizable(true);

		addWindowListener(new java.awt.event.WindowAdapter() {
			public void windowClosing(WindowEvent winEvt) {
				System.exit(0); // quick and dirty exit
			}
		});

		demoThread = new DemoThread();
		demoThread.start();

	}

	public void paint(java.awt.Graphics g) {
		BufferedImage[] imgCache;
		imgCache = imgs;
		if(imgCache != null) {
			int y =50;
			for(BufferedImage bi : imgCache) {
				g.drawImage(bi,0,y,null);
				y+= bi.getHeight() + 10;
			}
		}

	}

	class DemoThread extends Thread {

		public void run() {
			while(true) {
				try {
					imgs = vidProcessor.getMosaics(); // threading is taken care of in vidProcesor
					repaint();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
	}

	public static void main(String[] arg) {
		if(arg.length > 0) {
			ElProps.init(arg[0]);
		} else {
			ElProps.init("LAFace.props");
		}

		LAFaceVideoProcessor lafvp = new LAFaceVideoProcessor(ElProps.THE_PROPS);


		lafvp.setBackgroundAdaptation(ElProps.THE_PROPS.setProperty("adaptation", .1));

		try {

			lafvp.setSourceStream(ElProps.THE_PROPS.getProperty("camera", "axis"));
		} catch (IOException e) {
			e.printStackTrace();
		}


		lafvp.start();
		Demo demo = new Demo("Demo", lafvp);


	}

}
