package net.electroland.indy.test;

import java.text.ParseException;
import java.util.ArrayList;

/**
 * Some utility functions
 * @author geilfuss
 */
public class Util {

	/** TEST
	 * Given an array of bytes, returns a nicely formatted string of space
	 * delimited hex values representing the byte array.
	 * 
	 * @param b
	 * @return
	 */
	public static String bytesToHex(byte[] b){
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i< b.length; i++){
			sb.append(Integer.toHexString((b[i]&0xFF) | 0x100).substring(1,3) + " ");
		}
		return sb.toString();
	}

	public static String byteToHex(byte b){
		return Integer.toHexString((b&0xFF) | 0x100).substring(1,3);
	}
	
	/**
	 * Given a string of hex values between 0-255, generates a byte array.
	 * This method assumes you MUST use two hex values to define each byte, 
	 * so the string must have an even number of characters.
	 * 
	 * @param s
	 * @return
	 * @throws ParseException
	 * @throws NumberFormatException
	 */
	public static byte[] hextToBytes(String s) throws ParseException, NumberFormatException{
		ArrayList<Byte> a = new ArrayList<Byte>();		

		int l = s.length();
		if (l%2 == 0){
			for (int i = 0; i < l; i+=2){
				a.add((byte)Integer.parseInt(s.substring(i, i+2), 16));
			}
		}else{
			throw new ParseException("Odd number of bytes",0);
		}
		byte[] b = new byte[a.size()];
		for (int i = 0; i < b.length; i++){
			b[i] = ((Byte)a.get(i)).byteValue();
		}
		return b;
	}

	/**
	 * A fucking garbage method of encoding the time.  Giving a long, it 
	 * simply encodes the long as the bytes representing the characters 0-9.
	 * 
	 * E.g., 0000000000012 (12 milliseconds since the epoch) translates into a 13
	 * byte array, with the 12th and 13th bytes being 1 and 2.  We're doing this
	 * because our protocol doesn't let us use FE or FF in any packet.
	 * 
	 * @param t
	 * @return
	 */
	public static byte[] encodeTime(long t){
		byte[] bytes = ("" + t).getBytes();
		for (int i = 0; i< bytes.length; i++){
			bytes[i] = (byte)((int)bytes[i] - 48);
 		}
		return bytes;
	}

	/**
	 * Decodes anything encoded in encodeTime (above).  Returns -1 if the
	 * decoding fails.
	 */
	public static long decodeTime(byte[] bytes){
		long t = 0;
		for (int i= 0; i < bytes.length; i++){
			int b = (int)bytes[i];
			if (b > -1 && b < 10){
				t += (Math.pow(10, bytes.length - i -1) * b);
			}else{
				return -1;
			}
		}
		return t;
	}
}