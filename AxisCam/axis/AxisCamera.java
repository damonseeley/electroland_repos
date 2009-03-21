package axis;
/*
 * AxisCamera.java
 *
 * Created on March 12, 2004, 2:53 PM
 */

//obtained from URL: http://forum.java.sun.com/thread.jspa?threadID=494920&start=0&tstart=0

import java.net.*;
import com.sun.image.codec.jpeg.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import javax.swing.*;

/**
 *
 * @author David E. Mireles, Ph.D.
 * 
 * Modifications by Damon Seeley to use BufferedImage, Axis 211 compat, URL authorization
 */

public class AxisCamera implements Runnable {
	public boolean useMJPGStream = true;
	public String jpgURL;
	public String mjpgURL;
	DataInputStream dis;
	private BufferedImage image=null;
	public Dimension imageSize = null;
	public boolean connected = false;
	private boolean initCompleted = false;
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
	

	/** Creates a new instance of AxisCamera */
	/** baseURL,width,height,compression(0-100),color(0,1),username,password */
	public AxisCamera (String url, int w, int h, int comp, int color, String user, String pass) {
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
	}
	
//	 encodes username and password in Base64-encoding
	private String encodeUsernameAndPasswordInBase64( String usern, String psswd ) {
		    String s = usern + ":" + psswd;	    
		    String encs = new sun.misc.BASE64Encoder().encode(s.getBytes());	    
		    return "Basic " + encs;
	}
	

	public void connect(){
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
			if (!initCompleted) initDisplay();
		}catch(IOException e){ //incase no connection exists wait and try again, instead of printing the error
			try{
				huc.disconnect();
				Thread.sleep(60);
			}catch(InterruptedException ie){huc.disconnect();connect();}
			connect();
		}catch(Exception e){;}
	}

	public void initDisplay(){
		initCompleted = true;
	}

	public void disconnect(){
		try{
			if(connected){
				dis.close();
				connected = false;
			}
		}catch(Exception e){;}
	}

	public void readStream(){ //the basic method to continuously read the stream

		try{
			if (useMJPGStream){
				while(true){
					readMJPGStream();
				}
			}
			else{
				while(true){
					connect();
					readJPG();
					disconnect();
				}
			}
		}catch(Exception e){;}
	}


	
//	 Decode MJPG Stream for Axis 211 Camera version
	public void readMJPGStream(){
	readLine(4,dis); //discard the first 4 lines
	readJPG(); // Decode the JPEG image.
	readLine(1,dis); //discard the last line
	}

	public void readJPG(){ //read the embedded jpeg image
		try{
			
			JPEGImageDecoder decoder = JPEGCodec.createJPEGDecoder(dis);
			image = decoder.decodeAsBufferedImage();
			//System.out.println(image.getColorModel());
		}catch(Exception e){e.printStackTrace();disconnect();}
	}

	public void readLine(int n, DataInputStream dis){ //used to strip out the header lines
		for (int i=0; i<n;i++){
			readLine(dis);
		}
	}
	
	public void readLine(DataInputStream dis){
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
	
	public void run() {
		//TODO put a speedlimiter here
		connect();
		readStream();
	}

	// DS returns the current BufferedImage
	public BufferedImage getImage() {
		return image;
	}



}