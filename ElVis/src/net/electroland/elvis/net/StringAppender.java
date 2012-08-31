package net.electroland.elvis.net;

import java.util.StringTokenizer;

public interface StringAppender {
	public void buildString(StringBuilder sb);
	
	public static class EmptyStringAppender implements StringAppender {

		@Override
		public void buildString(StringBuilder sb) {
			
		}
		
		
	}
	
	public static class TrivialAppender implements StringAppender {
		String s ="";
		public void setString(String s) {
			this.s = s;
		}
		public void buildString(StringBuilder sb) {
			sb.append(s);
		}
	}

}
