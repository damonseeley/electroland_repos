package net.electroland.indy.test;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.StringTokenizer;

// this is just a convenient place to put an IP address + port.

public class Target {

	protected String ipstring;
	protected InetAddress address;
	protected int port;
	protected byte[] offset;
	private long lastHeardFrom = -1;

	public Target(String ipstring) throws IPAddressParseException, UnknownHostException{
		
		// Parse from "ipaddres:port".  We keep the original string this around
		// because InetAddress.toString() causes a DNS lookup.

		this.ipstring = ipstring;
		StringTokenizer st = new StringTokenizer(ipstring,":");

		address = InetAddress.getByName(st.nextToken());
		port = Integer.parseInt(st.nextToken());
		int offsetInt = Integer.parseInt(st.nextToken());
		
		if (offsetInt < 0 || offsetInt > 508){
			throw new IPAddressParseException("Offset must be between 0 and 508, inclusive");
		}

		offset = new byte[2];
		if (offsetInt > 253){
			offset[0] = (byte)253;
			offset[1] = (byte)(offsetInt - 253);
		}else{
			offset[0] = (byte)0;
			offset[1] = (byte)offsetInt;
		}
	}
	
	public InetAddress getAddress() {
		return address;
	}

	public int getPort() {
		return port;
	}

	public byte[] getOffset(){
		return offset;
	}
	
	public String toString(){
		return ipstring;
	}
	
	public void setLastHeardFrom(long l){
		this.lastHeardFrom = l;
	}
	public long getLastheardFrom(){
		return lastHeardFrom;
	}
}