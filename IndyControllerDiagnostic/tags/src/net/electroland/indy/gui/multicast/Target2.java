package net.electroland.indy.gui.multicast;

import java.net.InetSocketAddress;
import java.net.UnknownHostException;

import net.electroland.indy.test.IPAddressParseException;
import net.electroland.indy.test.Util;

/**
 * Does what InetSocketAddress does, plus stores an offset value. Our protocol 
 * stores offset as two bytes between 0 and 253, added together to specify an 
 * integer between 0 and 506.  Also, this obect caches the toString method,
 * since some versions of JAVA appear to do a host lookup when calling 
 * toString() on InetAddress objects.
 * 
 * @author geilfuss
 */
@SuppressWarnings("serial")
public class Target2 extends InetSocketAddress{

	protected String ipstring;
	protected byte[] offset;
	private long lastHeardFrom = -1;

	public Target2(String ip, int port) throws UnknownHostException{
		super(ip, port);

		System.out.println("will send to TCP " + this.toString());
	}
	
	public Target2(String ip, int port, int offsetInt) throws IPAddressParseException, UnknownHostException{

		super(ip, port);

		if (offsetInt < 0 || offsetInt > 506){
			throw new IPAddressParseException("Offset must be between 0 and 506, inclusive");
		}

		offset = new byte[2];
		if (offsetInt > 253){
			offset[0] = (byte)253;
			offset[1] = (byte)(offsetInt - 253);
		}else{
			offset[0] = (byte)0;
			offset[1] = (byte)offsetInt;
		}
		System.out.println("will send to UDP " + this.toString() + ':' + Util.bytesToHex(offset));
	}

	public byte[] getOffset(){
		return offset;
	}

	public String toString(){
		if (ipstring == null){
			StringBuffer sb = new StringBuffer();
			sb.append(getAddress()).append(':').append(getPort());
			ipstring = sb.toString();
		}
		return ipstring;
	}
	/**
	 * not used yet.
	 * @param l
	 */
	public void setLastHeardFrom(long l){
		this.lastHeardFrom = l;
	}
	/**
	 * not used yet.
	 * @return
	 */
	public long getLastheardFrom(){
		return lastHeardFrom;
	}
}