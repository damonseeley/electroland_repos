package net.electroland.util;

public class Util {

	public static byte getHiByte(short s){
		return (byte)(s >> 8);
	}
	
	public static byte getLoByte(short s){
		return (byte)s;
	}
	public static String bytesToHex(byte b)
	{
		return Integer.toHexString((b&0xFF) | 0x100).substring(1,3) + " ";
	}
	public static String bytesToHex(byte[] b, int length)
	{
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i< length; i++){
			sb.append(Integer.toHexString((b[i]&0xFF) | 0x100).substring(1,3) + " ");
		}
		return sb.toString();
	}
	public static int unsignedByteToInt(byte b) 
	{
		return (int) b & 0xFF;
	}
	public static int[] fitCurve(int[] in)
	{
		// doesn't do jack shit yet.
		return in;
	}
}