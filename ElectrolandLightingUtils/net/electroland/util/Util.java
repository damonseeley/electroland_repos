package net.electroland.util;

public class Util {

	public static byte getHiByte(short s){
		return (byte)(s >> 8);
	}
	
	public static byte getLoByte(short s){
		return (byte)s;
	}
}
