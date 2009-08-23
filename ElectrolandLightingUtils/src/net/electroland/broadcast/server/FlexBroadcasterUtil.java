package net.electroland.broadcast.server;

import java.util.Random;

public class FlexBroadcasterUtil {

	public static final String XML_HEADER = "<?xml version=\"1.0\" encoding=\"utf-8\"?>";

	private static Random r = new Random(System.currentTimeMillis());

	public static String getUniqueId(){
		// return a 20 character long random ascii string.
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < 20; i++){
			sb.append((char)(65 + r.nextInt(26)));
		}
		return sb.toString();
	}
}