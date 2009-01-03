package net.electroland.faces;

import java.util.ArrayList;

public class Util {

	public static Light[] getAllLights(LightController[] controllers){
		ArrayList<Light> a = new ArrayList<Light>();
		for (int i = controllers.length - 1; i >= 0; i--){
			a.addAll(controllers[i].lights);
		}
		Light[] l = new Light[a.size()];
		a.toArray(l);
		return l;
	}

	public static String bytesToHex(int i){
		return Integer.toHexString((((byte)i)&0xFF) | 0x100).substring(1,3);
	}
	
	public static String bytesToHex(byte[] b){
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i< b.length; i++){
			sb.append(Integer.toHexString((b[i]&0xFF) | 0x100).substring(1,3) + " ");
		}
		return sb.toString();
	}
}
