package net.electroland.elvis.blobtracking;

import java.util.StringTokenizer;

import net.electroland.elvis.net.StringAppender;

public class BaseTrack implements StringAppender{
	public int id;
	public float x;
	public float y;
	public boolean isProvisional = true;
	
	public BaseTrack(int id) {
		this.id = id;
	}
	
	
	public BaseTrack(int id, float x, float y, boolean isProv) {
		this.id = id;
		this.x = x;
		this.y = y;
		this.isProvisional = isProv;
	}
	
	public String toString() {
		return "Track:" + id + " (" + x +" ," + y +")"; 
	}

	
	public int getId() {
		return id;
	}
	public float getX() {
		return x;
	}
	public float getY() {
		return y;
	}
	public boolean isProvisional() {
		return isProvisional;
	}

	// id, x, y, provisional

		public void buildString(StringBuilder sb) {
			sb.append(id);
			sb.append(",");
			sb.append(x);
			sb.append(",");
			sb.append(y);
			sb.append(",");
			sb.append(isProvisional);
		}
		
		public static BaseTrack buildFromTokenizer(StringTokenizer tokenizer) {
			if (! tokenizer.hasMoreTokens()) {
				return null;
			}
			String token = tokenizer.nextToken();
			
			if (token.equals("|")) {
				return null;
			} 
			int id = Integer.parseInt(token);
			float x = Float.parseFloat(tokenizer.nextToken());
			float y = Float.parseFloat(tokenizer.nextToken());
			boolean b = Boolean.parseBoolean(tokenizer.nextToken());
			return new BaseTrack(id,x,y,b);
			
		}

}
