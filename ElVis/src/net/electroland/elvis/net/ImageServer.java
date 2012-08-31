package net.electroland.elvis.net;

import static com.googlecode.javacv.cpp.opencv_imgproc.CV_INTER_AREA;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvResize;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Timer;
import java.util.TimerTask;

import net.electroland.elvis.imaging.PresenceDetector;

import com.googlecode.javacv.cpp.opencv_core.IplImage;


public class ImageServer extends Thread {
	ServerSocket socket;
	boolean isRunning = true;
	BufferedImage img;
	DataOutputStream outputStream;
	PresenceDetector pd; 

	Timer timer;

	BufferedImage scaledImage;


	int width;
	int height;

	public ImageServer(PresenceDetector pd, int port) throws IOException {
		super();
		this.pd = pd;
		socket = new ServerSocket(port);
	}



	public void interpretCommand(String s) {
		System.out.println("interpreting " + s);
		try {
			String[] command = s.split(",");
			if(s.equals("stop")) {
				if(timer != null) timer.cancel();
			} else if (s.equals("close")) {
				if(timer != null) timer.cancel();
				try {
					socket.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			} else if(command.length == 4) {
				if(timer != null) timer.cancel();
				pd.setNetImageReturn(command[0]);
				width = Integer.parseInt(command[1]);
				height = Integer.parseInt(command[2]);
				scaledImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
				float fps = Float.parseFloat(command[3]);
				long period =(long) ( 1000 / fps);
				timer = new Timer();
				timer.scheduleAtFixedRate(new ImageSender(), period, period);	
			} 
		} catch (RuntimeException e) {
			System.out.println("Exception while parsing " + s);
			e.printStackTrace();

		}
	}

	public void run() {
		while(isRunning) {
			Socket connectionSocket;
			try {
				connectionSocket = socket.accept();
				outputStream = new DataOutputStream(connectionSocket.getOutputStream());
				BufferedReader reader = new BufferedReader(new InputStreamReader(connectionSocket.getInputStream()));
				String inputLine = null;
				while ((inputLine = reader.readLine()) != null) {
					interpretCommand(inputLine);
				}
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

	}
	public class ImageSender extends TimerTask {

		@Override
		public void run() {
			IplImage img = pd.getNetImage();
			if(img != null) {
				IplImage scaledImage = IplImage.create(width, height, img.depth(),1);
				cvResize(img, scaledImage, CV_INTER_AREA);
				Raster raster = scaledImage.getBufferedImage().getRaster();
				int[] pixels = new int[width*height];
				pixels = raster.getPixels(0, 0, width, height, pixels);
				for(int i = 0; i < pixels.length; i++) {
//					System.out.println("sending " + pixels[i]);
					try {
//						System.out.println("sending"+ pixels[i]);
						outputStream.write(pixels[i]);
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
		}
	}
}



