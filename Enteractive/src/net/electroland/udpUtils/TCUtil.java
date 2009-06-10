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
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import net.electroland.enteractive.core.Tile;
import net.electroland.enteractive.core.TileController;
import net.electroland.enteractive.utils.HexUtils;

public class TCUtil {

	public Properties tileProps;
	private List<TileController> tileControllers;
	private DatagramSocket socket;
	private String startByte, endByte, updateByte, feedbackByte;
	private String onChangeByte, powerByte, reportByte, offsetByte, mcResetByte;
	private int tileTimeout = 30000;		// tiles are rebooted after this duration of being on
	private int powerCycleDuration = 300;	// duration to keep tile off when cycled
	
	public TCUtil(){
		try{
			tileProps = new Properties();
			tileProps.load(new FileInputStream(new File("depends//tile.properties")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
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
	
	/*
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
				if(tile.getSensorState() && tile.getAge() > tileTimeout && !tile.rebooting){
					//System.out.println("tile "+tile.getID()+" on too long");
					triggerStateChange = true; 		// turn power off for this one tile
					tile.reboot();
					powerStates[i] = false;
				} else if(tile.rebooting && tile.offPeriod() > powerCycleDuration){
					triggerStateChange = true;		// turn power back on
					powerStates[i] = true;
					tile.rebooting = false;
				} else if(tile.rebooting){
					powerStates[i] = false;			// leave power off for this one tile
				} else {
					powerStates[i] = true;			// leave power on for this one tile
				}
				i++;
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
				byte[] buf = HexUtils.hexToBytes(startByte +" "+ powerByte + payload + "00 " + Integer.toHexString(tc.getOffset()-1) +" "+ endByte);
				//HexUtils.printHex(buf);		// for debugging
				send(tc.getAddress(), buf);
			}
		}
	}
	
	*/
	
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
	
	public void billyJeanMode(){
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
