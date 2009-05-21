package axis;
/*
 * AxisCamera.java
 *
 * Created on March 12, 2004, 2:53 PM
 */

//obtained from URL: http://forum.java.sun.com/thread.jspa?threadID=494920&start=0&tstart=0

import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Image;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

import javax.swing.JFrame;
import javax.swing.JPanel;

import com.sun.image.codec.jpeg.JPEGCodec;
import com.sun.image.codec.jpeg.JPEGImageDecoder;

/**
 *
 * @author David E. Mireles, Ph.D.
 */
public class AxisCameraTestApp extends JPanel implements Runnable {
	public boolean useMJPGStream = true;
	public String jpgURL="http://11flower.dyndns.org/axis-cgi/jpg/image.cgi?resolution=320x240";
	public String mjpgURL="http://11flower.dyndns.org/axis-cgi/mjpg/video.cgi?resolution=160x120&compression=0";
	DataInputStream dis;
	private Image image=null;
	public Dimension imageSize = null;
	public boolean connected = false;
	private boolean initCompleted = false;
	HttpURLConnection huc=null;
	Component parent;
	
	
//	 username and password for the user who has explicitly been added to the user-list of the camera by the admin
	String username = "root";
	String password = "11fl0wer";
	String base64authorization = null;

	//my modifications below
	public int imageMag = 4;

	/** Creates a new instance of AxisCamera */
	public AxisCameraTestApp (Component parent_) {
		parent = parent_;
//		 only use authorization if all informations are available
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
//			System.out.println(huc.getContentType());
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

	public void initDisplay(){ //setup the display
		if (useMJPGStream)readMJPGStream();
		else {readJPG();disconnect();}
		imageSize = new Dimension(image.getWidth(this)*imageMag, image.getHeight(this)*imageMag+20);
		setPreferredSize(imageSize);
		parent.setSize(imageSize);
		parent.validate();
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

	public void paint(Graphics g) { //used to set the image on the panel
		if (image != null)
			g.drawImage(image, 0, 0, image.getWidth(this)*imageMag,image.getHeight(this)*imageMag,this);
	}

	public void readStream(){ //the basic method to continuously read the stream
		System.out.println("reading stream");
		try{
			if (useMJPGStream){
				while(true){
					readMJPGStream();
					parent.repaint();
				}
			}
			else{
				while(true){
					connect();
					readJPG();
					parent.repaint();
					disconnect();

				}
			}

		}catch(Exception e){;}
	}


	
//	 Decode MJPG Stream for Axis 213 PTZ Camera version 4.12 - BRAIN STONE
	public void readMJPGStream(){
	readLine(4,dis); //discard the first 4 lines
	readJPG(); // Decode the JPEG image.
	readLine(1,dis); //discard the last line
	}

	public void readJPG(){ //read the embedded jpeg image
		try{
			JPEGImageDecoder decoder = JPEGCodec.createJPEGDecoder(dis);
			image = decoder.decodeAsBufferedImage();
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
//				System.out.print(t); //uncomment if you want to see what the lines actually look like
				if(t.equals(lineEnd)) end=true;
			}
		}catch(Exception e){e.printStackTrace();}


	}
	public void run() {
		connect();
		readStream();
		System.out.println("running");
	}


	public static void main(String[] args) {
		JFrame jframe = new JFrame();
		jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		AxisCameraTestApp axPanel = new AxisCameraTestApp(jframe);
		new Thread(axPanel).start();
		jframe.getContentPane().add(axPanel);
		jframe.pack();
		jframe.setVisible(true);
	}


}