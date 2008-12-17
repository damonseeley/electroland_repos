package net.electroland.connection.core;

import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.UnknownHostException;

/**
 * Used for interfacing with individual lighting controllers in order to
 * keep track of their status and maintain the proper offsets.
 */

public class LightBox {
	
	public int id;			// ID number
	public InetAddress ip;	// controller IP address
	public int offsetA;	// A side offset
	public int offsetB;	// B side offset
	DatagramSocket socket;
	
	
	public LightBox(int _id, String _ip, int _offset){
		id = _id;
		try{
			ip = InetAddress.getByName(_ip);
		} catch(UnknownHostException e){
			e.printStackTrace();
		}
		offsetA = _offset;
		offsetB = _offset + 12;
	}
}
