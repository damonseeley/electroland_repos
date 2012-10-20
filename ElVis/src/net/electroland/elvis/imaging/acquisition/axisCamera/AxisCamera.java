package net.electroland.elvis.imaging.acquisition.axisCamera;
/*
 * AxisCamera.java
 *
 * Created on March 12, 2004, 2:53 PM
 */

//obtained from URL: http://forum.java.sun.com/thread.jspa?threadID=494920&start=0&tstart=0

import java.awt.Component;
import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.ImageReceiver;

import com.sun.image.codec.jpeg.ImageFormatException;
import com.sun.image.codec.jpeg.JPEGCodec;
import com.sun.image.codec.jpeg.JPEGImageDecoder;

/**
 *
 * @author David E. Mireles, Ph.D.
 * 
 * Modifications by Damon Seeley to use BufferedImage, Axis 211 compat, URL authorization
 * Changed to use a synchornizedImage to make thread safe
 */

public class AxisCamera extends Thread implements ImageAcquirer {
	boolean isRunning;
	public boolean useMJPGStream = true;
	public String jpgURL;
	public String mjpgURL;
	DataInputStream dis;
	public Dimension imageSize = null;
	public boolean connected = false;
	HttpURLConnection huc=null;
	Component parent;

	String username;
	String password;
	String base64authorization = null;

	public String baseURL;
	public int w;
	public int h;
	public int compression;
	public int color;

	public static final int MAX_CONNECTION_ATTEMPTS = 5;

	ImageReceiver imageReceiver;

	/* Creates a new instance of AxisCamera */
	/* baseURL,width,height,compression(0-100),color(0,1),username,password */
	// imageReceiver can be null if not using readStream



	public AxisCamera (String url, int w, int h, int comp, int color, String user, String pass, ImageReceiver imageReceiver) {
		this.w = w;
		this.h = h;
		this.baseURL = url;
		this.compression = comp;
		this.color = color;
		this.username = user;
		this.password = pass;

		jpgURL=baseURL + "axis-cgi/jpg/image.cgi?resolution=" + w + "x" + h + "&compression=" + compression + "&color=" + color;
		mjpgURL=baseURL + "axis-cgi/mjpg/video.cgi?resolution=" + w + "x" + h + "&compression=" + compression + "&color=" + color;


		if(username == null || password == null) {
			System.out.println("not using authorization");
		} else {
			base64authorization = this.encodeUsernameAndPasswordInBase64(username, password);
		}

		this.imageReceiver = imageReceiver;

	}

//	encodes username and password in Base64-encoding
	private String encodeUsernameAndPasswordInBase64( String usern, String psswd ) {
		String s = usern + ":" + psswd;	    
		String encs = new sun.misc.BASE64Encoder().encode(s.getBytes());	    
		return "Basic " + encs;
	}

	protected void connect() throws ElCameraConnectionException {
		connect(MAX_CONNECTION_ATTEMPTS);
	}

	protected void connect(int connectAttemps) throws ElCameraConnectionException{
		if(connectAttemps < 0) {
			URL u;
			try {
				u = new URL(useMJPGStream?mjpgURL:jpgURL);
				throw new ElCameraConnectionException("Unable to connect to " + u);
			} catch (MalformedURLException e) {
				throw new ElCameraConnectionException("Unable to connect to axis cam URL may be malformed");
			}
		}
		try{
			URL u = new URL(useMJPGStream?mjpgURL:jpgURL);
			huc = (HttpURLConnection) u.openConnection();
			// if authorization is required set up the connection with the encoded authorization-information			
			if(base64authorization != null) {
				huc.setDoInput(true);
				huc.setRequestProperty("Authorization",base64authorization);
				huc.connect();
			}
			InputStream is = huc.getInputStream();
			connected = true;
			BufferedInputStream bis = new BufferedInputStream(is);
			dis= new DataInputStream(bis);
		}catch(IOException e){ //incase no connection exists wait and try again, instead of printing the error
			try{
				huc.disconnect();
				Thread.sleep(60);
			}catch(InterruptedException ie){
				huc.disconnect();
				connect(connectAttemps -1);
			}
			connect(connectAttemps-1);
		}catch(Exception e){
			imageReceiver.receiveErrorMsg(e);
		}
	}


	protected void disconnect(){
		try{
			if(connected){
				dis.close();
				connected = false;
			}
			System.out.println(mjpgURL);

		}catch(Exception e){;}
	}

	protected void readStream(){ //the basic method to continuously read the stream

		try{
			if (useMJPGStream){
				connect();
				while(isRunning){
					readMJPGStream();
				}
				disconnect();
			}
			else{
				while(isRunning){
					connect();
					imageReceiver.addImage(readJPG());
					disconnect();
				}
			}
		}catch(Exception e){
			imageReceiver.receiveErrorMsg(e);
		}
	}



//	Decode MJPG Stream for Axis 211 Camera version
	public void readMJPGStream() throws ElCameraConnectionException{
		readLine(4,dis); //discard the first 4 lines
		try {
			imageReceiver.addImage(readJPG());
			readLine(1,dis); //discard the last line
		} catch (IOException e) {
			e.printStackTrace();
			connect();

		} // Decode the JPEG image.
	}

	public BufferedImage readJPG() throws IOException { //read the embedded jpeg image

		JPEGImageDecoder decoder = JPEGCodec.createJPEGDecoder(dis);

		try {
			return decoder.decodeAsBufferedImage();
		} catch (ImageFormatException e) {
			disconnect();
			throw new IOException(e.toString());
		} catch (IOException e) {
			disconnect();
			throw e;
		} 

	}

	protected void readLine(int n, DataInputStream dis){ //used to strip out the header lines
		for (int i=0; i<n;i++){
			readLine(dis);
		}
	}

	protected void readLine(DataInputStream dis){
		try{
			boolean end = false;
			String lineEnd = "\n"; //assumes that the end of the line is marked with this
			byte[] lineEndBytes = lineEnd.getBytes();
			byte[] byteBuf = new byte[lineEndBytes.length];

			while(!end){
				dis.read(byteBuf,0,lineEndBytes.length);
				String t = new String(byteBuf);
				//System.out.print(t); //uncomment if you want to see what the lines actually look like
				if(t.equals(lineEnd)) end=true;
			}
		}catch(Exception e){e.printStackTrace();}
	}

	public void stopRunning() {
		isRunning = false;
	}

	public void run() {
		isRunning = true;
		readStream();
	}

	public void stopRunningForce() {
		stopRunning();
		
	}




}