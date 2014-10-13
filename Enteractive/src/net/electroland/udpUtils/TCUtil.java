package net.electroland.udpUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import net.electroland.enteractive.core.Tile;
import net.electroland.enteractive.core.TileController;
import net.electroland.enteractive.utils.HexUtils;

import org.apache.log4j.Logger;

public class TCUtil {

	static Logger logger = Logger.getLogger(TCUtil.class);
	static Logger tileLogger = Logger.getLogger("TileErrors");
	
	public Properties tileProps;
	private List<TileController> tileControllers;
	private DatagramSocket socket;
	private String startByte, endByte, updateByte, feedbackByte;
	private String onChangeByte, powerByte, reportByte, offsetByte, mcResetByte;
	
	//private int tileTimeout = 120000;		// tiles are rebooted after this duration of being on
	//2014 testing value
	private int tileTimeout = 8000;		// tiles are rebooted after this duration of being on

	private int powerCycleDuration = 120000;	// duration to keep tile off when cycled
	
	public TCUtil(int timeout){
		try{
			tileProps = new Properties();
			tileProps.load(new FileInputStream(new File("depends//tile.properties")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		tileTimeout = timeout;
		logger.info("Tile timeout set to " + tileTimeout/1000 + " seconds");
		
		startByte = tileProps.getProperty("startByte");
		endByte = tileProps.getProperty("endByte");
		updateByte = tileProps.getProperty("updateByte");
		feedbackByte = tileProps.getProperty("feedbackByte");
		onChangeByte = tileProps.getProperty("onChangeByte");
		powerByte = tileProps.getProperty("powerByte");
		reportByte = tileProps.getProperty("reportByte");
		offsetByte = tileProps.getProperty("offsetByte");
		mcResetByte = tileProps.getProperty("mcResetByte");
		
		tileControllers = new ArrayList<TileController>();						// instantiate tileControllers list
		Iterator<Map.Entry<Object, Object>> iter = tileProps.entrySet().iterator();			
		while(iter.hasNext()){													// for each property...
			Map.Entry<Object, Object> pair = (Map.Entry<Object, Object>)iter.next();
			if(pair.getKey().toString().startsWith("TileController")){			// if it's a tile controller...
				String[] values = pair.getValue().toString().split(",");		// add new TileController object with arguments
				tileControllers.add(new TileController(Integer.parseInt(pair.getKey().toString().substring(14)), values[0], Integer.parseInt(values[1]), Integer.parseInt(values[2])));
			}
		}
		
		try {
			socket = new DatagramSocket();
		} catch (SocketException e) {
			e.printStackTrace();
		}
		
		sendOffsetPackets();
	}
	
	public Map<Integer, Tile> getStuckTiles(){
		HashMap<Integer, Tile> stuckTiles = new HashMap<Integer, Tile>();
		for (TileController tc : tileControllers){
			for (Tile t : tc.getTiles()){
				if (t.stuck){
					stuckTiles.put(t.getID(), t);
				}
			}
		}
		return stuckTiles;
	}
	
	// TODO THIS IS WHAT ENDED UP "BREAKING" TILES ON SITE
	
	public void checkTileStates(){
		Iterator<TileController> iter = tileControllers.iterator();
		while(iter.hasNext()){						// check every tile controller
			TileController tc = iter.next();
			List<Tile> tiles = tc.getTiles();
			Iterator<Tile> tileiter = tiles.iterator();
			boolean[] powerStates = new boolean[tiles.size()];
			boolean triggerStateChange = false;
			int i=0;
			while(tileiter.hasNext()){
				Tile tile = tileiter.next();
				
				// TODO - CHECK TILE AGE AND ALSO IF TILE HAS COME "UNSTUCK"
				
				
				if(tile.getSensorState() && tile.getAge() > tileTimeout){
					if (!tile.getStuck()) {
						tileLogger.info("TILE "+tile.getID() + " IS STUCK on TileController #" + tc.getID() + ", IP=" + tc.getAddress());
					}
					tile.setStuck(true);
					
				} else if (!tile.getSensorState()) {
					if (tile.getStuck()){
						tileLogger.info("TILE "+tile.getID()+ " CAME UNSTUCK");
					}
					tile.setStuck(false);
				}
				
				/* 2014 This is the former logic for tile 'rebooting' which is now not used
				 * due to potential power faults caused by reboot states on a controller
				 * note some of this logic has been duplicated above to check
				 * for tile 'stuck' and 'unstuck' cases

				if(tile.getSensorState() && tile.getAge() > tileTimeout && !tile.rebooting){
					tileLogger.info("stuck tile,"+tile.getID()+ ","+ tileTimeout/1000 + "seconds");
					grabWebcamImage();
					//triggerStateChange = true; 		// turn power off for this one tile
					tile.reboot();
					powerStates[i] = false;
				} else if(tile.rebooting && tile.offPeriod() > powerCycleDuration){
					//triggerStateChange = true;		// turn power back on
					powerStates[i] = true;
					tile.rebooting = false;
				} else if(tile.rebooting){
					powerStates[i] = false;			// leave power off for this one tile
				} else {
					powerStates[i] = true;			// leave power on for this one tile
				}
				
				*/
				
				i++;  // 2014 what is this doing, exactly?
			
			}
			
			if(triggerStateChange){					// cause the actual power cycle here
				//System.out.println("triggering a state change");
				String payload = " ";
				for(int n=0; n<powerStates.length; n++){
					if(powerStates[n]){
						payload += "FD ";
					} else {
						payload += "00 ";
					}
				}
				// TODO DO NOT UNCOMMENT THIS ON SITE!!!!  MIGHT KILL TILES!!!
				//byte[] buf = HexUtils.hexToBytes(startByte +" "+ powerByte + payload + "00 " + Integer.toHexString(tc.getOffset()-1) +" "+ endByte);
				//HexUtils.printHex(buf);		// for debugging
				//SND(tc.getAddress(), buf);  //purposefully mis-addressed this call to prevent accidental firing.  I am paranoid.
			}
		}
	}
	
	public void grabWebcamImage(){
		
		// 2014 disable all this webcam nonsense, no longer works
		/*
		String url = "http://11flower.dyndns.org/axis-cgi/io/virtualinput.cgi?action=6:/";
		String s = "stucktile:stucktile";	    
		String base64authorization = "Basic " + new sun.misc.BASE64Encoder().encode(s.getBytes());	    
		try{
			URL u = new URL(url);
			HttpURLConnection huc = (HttpURLConnection) u.openConnection();
			huc.setDoInput(true);
			huc.setRequestProperty("Authorization",base64authorization);
			huc.connect();		// just connecting to trigger image request
			//InputStream is = new InputStream();
			huc.getInputStream();
			huc.disconnect();
			//logger.info("STUCKTILE: webcam event triggered");
		} catch (IOException e){
			logger.info(e);
			logger.info("stuck tile: unable to access webcam image");
		}
		*/
	}
	
	
	public List<TileController> getTileControllers(){
		return tileControllers;
	}
	
	public void updateLights(int offset, byte[] data){
		int[] values = new int[data.length];
		for(int i=0; i<data.length; i++){
			values[i] = (int)(data[i] & 0xFF);
		}
		Iterator<TileController> iter = tileControllers.iterator();
		while(iter.hasNext()){
			TileController tc = iter.next();
			if(tc.getOffset() == offset){
				tc.setLightValues(values);
			}
		}
	}
	
	public void updateSensors(int offset, byte[] data){
		boolean[] newdata = new boolean[data.length];
		for(int i=0; i<data.length; i++){
			if((int)(data[i] & 0xFF) > 0){
				newdata[i] = true;
			} else {
				newdata[i] = false;
			}
		}
		//System.out.println("update sensors at "+ offset);
		Iterator<TileController> iter = tileControllers.iterator();
		while(iter.hasNext()){
			TileController tc = iter.next();
			//System.out.println(tc.getOffset() +" "+ offset);
			if(tc.getOffset()-1 == offset){
				tc.setSensorStates(newdata);
			}
		}
	}
	
	public void turnOffAllTiles(){
		Iterator<TileController> iter = tileControllers.iterator();
		while(iter.hasNext()){
			TileController tc = iter.next();
			byte[] buf = HexUtils.hexToBytes(startByte +" "+ powerByte +" 00 00 00 00 00 00 00 00 00 "+ Integer.toHexString(tc.getOffset()-1) +" "+ endByte);
			send(tc.getAddress(), buf);
		}
	}
	
	public void turnOnAllTiles(){
		Iterator<TileController> iter = tileControllers.iterator();
		while(iter.hasNext()){
			TileController tc = iter.next();
			byte[] buf = HexUtils.hexToBytes(startByte +" "+ powerByte +" FD FD FD FD FD FD FD FD 00 "+ Integer.toHexString(tc.getOffset()-1) +" "+ endByte);
			send(tc.getAddress(), buf);
		}
	}
	
	public void resetAllTCs(){
		Iterator<TileController> iter = tileControllers.iterator();
		while(iter.hasNext()){
			TileController tc = iter.next();
			resetTileController(tc);
		}
	}
	
	public void resetTileController(TileController tc){
		byte[] buf = HexUtils.hexToBytes(startByte +" "+ mcResetByte +" AA 55 "+ endByte);	// from 1-based to 0-based
		send(tc.getAddress(), buf);
		System.out.println("MC reset packet sent to TC "+tc.getID());
	}
	
	public void sendOffsetPackets(){
		Iterator<TileController> iter = tileControllers.iterator();
		while(iter.hasNext()){
			TileController tc = iter.next();
			sendOffsetPacket(tc);
		}
	}
	
	public void sendOffsetPacket(TileController tc)
	{
		int offset1 = tc.getOffset() - 1;
		int offset2 = 0;
		if (offset1 > 253)
		{
			offset2 = offset1 - 253;
			offset1 = 253;
		}
		byte[] buf = new byte[5];
		buf[0] = (byte)255;
		buf[1] = (byte)128;
		buf[2] = (byte)offset1;
		buf[3] = (byte)offset2;
		buf[4] = (byte)254;
		//System.out.println(HexUtils.bytesToHex(buf, buf.length));
		send(tc.getAddress(), buf);
	}
	
	public void send(String ip, byte[] buf){
		InetAddress address = null;
		try {
			address = InetAddress.getByName(ip);
		} catch (UnknownHostException e1) {
			e1.printStackTrace();
		}
		int port = 10001;
		DatagramPacket packet = new DatagramPacket(buf, buf.length, address, port);
		try {
			socket.send(packet);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
}
