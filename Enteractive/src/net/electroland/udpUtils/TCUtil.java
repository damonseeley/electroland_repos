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
		
		tileControllers = new ArrayList<TileController>();			// instantiate tileControllers list
		Iterator<Map.Entry<Object, Object>> iter = tileProps.entrySet().iterator();			
		while(iter.hasNext()){										// for each property...
			Map.Entry<Object, Object> pair = (Map.Entry<Object, Object>)iter.next();
			if(pair.getKey().toString().startsWith("TileController")){			// if it's a tile controller...
				String[] values = pair.getValue().toString().split(",");		// add new TileController object with arguments
				tileControllers.add(new TileController(Integer.parseInt(pair.getKey().toString().substring(14)), values[0], Integer.parseInt(values[1]), Integer.parseInt(values[2])));
			}
		}
		iter.remove();
		
		try {
			socket = new DatagramSocket();
		} catch (SocketException e) {
			e.printStackTrace();
		}
		
		sendOffsetPackets();
	}
	
	public void turnOffTile(Tile tile){
		// TODO cycle power off
	}
	
	public void turnOnTile(Tile tile){
		// TODO cycle power on
	}
	
	public void turnOffAllTiles(){
		Iterator<TileController> iter = tileControllers.iterator();
		while(iter.hasNext()){
			TileController tc = iter.next();
			byte[] buf = HexUtils.hexToBytes(startByte +" "+ powerByte +" 00 00 00 00 00 00 00 00 00 "+ Integer.toHexString(tc.getOffset()-1) +" "+ endByte);
			send(tc.getAddress(), buf);
		}
		iter.remove();
	}
	
	public void turnOnAllTiles(){
		Iterator<TileController> iter = tileControllers.iterator();
		while(iter.hasNext()){
			TileController tc = iter.next();
			byte[] buf = HexUtils.hexToBytes(startByte +" "+ powerByte +" FD FD FD FD FD FD FD FD 00 "+ Integer.toHexString(tc.getOffset()-1) +" "+ endByte);
			send(tc.getAddress(), buf);
		}
		iter.remove();
	}
	
	public void billyJeanMode(){
		Iterator<TileController> iter = tileControllers.iterator();
		while(iter.hasNext()){
			TileController tc = iter.next();
			resetTileController(tc);
		}
		iter.remove();
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
		iter.remove();
	}
	
	public void sendOffsetPacket(TileController tc){
		byte[] buf = HexUtils.hexToBytes(startByte +" "+ offsetByte +" 00 "+ Integer.toHexString(tc.getOffset()-1) +" "+ endByte);	// from 1-based to 0-based
		send(tc.getAddress(), buf);
		//System.out.println("offset packet sent to TC "+tc.getID());
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
