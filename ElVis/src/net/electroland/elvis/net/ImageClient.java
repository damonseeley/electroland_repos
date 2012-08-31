package net.electroland.elvis.net;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;

public abstract class ImageClient extends Thread {
	boolean isRunning = true;

	Socket socket;
	PrintWriter out;
	InputStream in;
	int width;
	int height;


	public ImageClient(String address, int port) throws UnknownHostException, IOException {
		super();
		socket = new Socket(address, port);
		out = new PrintWriter(socket.getOutputStream(), true);
		in =socket.getInputStream();	
	}

	public void setMode(String mode, int width, int height, float fps) throws IOException {
		this.width = width;
		this.height = height;
		out.print(mode);
		out.print(",");
		out.print(Integer.toString(width));
		out.print(",");
		out.print(Integer.toString(height));
		out.print(",");
		out.println(Float.toString(fps));
	}

	public void run () {
		while(isRunning) {
			try {
				//byte[] pixels = new byte[width*height];
				//				for(int i = 0; i < pixels.length ; i++) {
				//				
				//			pixels[i] = (byte) in.read();
				//	}
				BufferedImage scaledImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
				for(int y = 0; y < height; y++) {
					for(int x = 0; x < width; x++) {
						int gray = in.read();
						int rgbVal = (gray << 16) + (gray << 8) + (gray); 
						scaledImage.setRGB(x, y, rgbVal);
					}
				}
				//				WritableRaster raster = (WritableRaster) scaledImage.getData();
				//				raster.setDataElements(0,0,width,height,pixels);
				//				scaledImage.getGraphics().setColor(new Color());
				//				scaledImage.getGraphics().drawLine(0, 0, width, height);
				handelImage(scaledImage);

			} catch (IOException e) {
				e.printStackTrace();
			} catch (RuntimeException e) {
				e.printStackTrace();				
			}
		}

	}

	public abstract void handelImage(BufferedImage img);


}
