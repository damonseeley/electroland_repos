package net.electroland.elvis.blobktracking.core;

import java.awt.Graphics;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.UnknownHostException;

import javax.swing.JFrame;

import net.electroland.elvis.net.ImageClient;
import net.electroland.elvis.util.ElProps;

public class NetworkImageViewer extends ImageClient {
	ImageFrame frame;

	public NetworkImageViewer(ElProps props) throws UnknownHostException, IOException {
		this(	props.getProperty("imageServerAddress", "localhost"),
				props.getProperty("imageServerPort", 3598));		
	}
	public NetworkImageViewer(String address, int port) throws UnknownHostException, IOException {
		super(address, port);
		frame = new ImageFrame("Listening to " + address);
		frame.addWindowListener(new WindowAdapter()
		{
			public void windowClosing(WindowEvent e)
			{
				stopRunning();
			}
		});

		start();
	}

	@Override
	public void handelImage(BufferedImage img) {
		frame.setImage(img);

	}

	public class ImageFrame extends JFrame {
		BufferedImage img = null;
		public ImageFrame(String name) {
			super(name);
			setSize(640, 480);
			setVisible(true);			
		}
		public void paint(Graphics g) {
			if(img != null) {
				g.drawImage(img, 0, 0, frame.getWidth(), frame.getHeight(), frame);
			}
		}
		public void setImage(BufferedImage img) {
			this.img = img;
			frame.repaint();
		}
	}
	public static void main(String[] args) throws UnknownHostException, IOException {
		ElProps props;
		if(args.length > 0) {
			props = ElProps.init(args[0]);
		} else {
			props =ElProps.init("blobTracker.props");
		}

		NetworkImageViewer niv = new NetworkImageViewer( props );
		niv.setMode(props.getProperty("imageServerType","RAW"),
				props.getProperty("imageServerWidth",160),
				props.getProperty("imageServerheight",120),
				props.getProperty("imageServerFPS",120));


	}

}
