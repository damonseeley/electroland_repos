package net.electroland.noho.core.fontMatrix;

import java.util.Vector;

import net.electroland.noho.core.NoHoConfig;

public class LineFormatter {
	public static final String LINEBREAK = "\n";


	static int maxCharsPerLine = NoHoConfig.DISPLAYWIDTH / NoHoConfig.CHARWIDTH; 

	//TODO: do we need padding? (IE do charaters look ok flush left and right?)



	//TODO: probably should add code to check for unrecognized chars
	public static Vector<String> formatString(String string) {
		Vector<String> lines = new Vector<String>();
		String[]  rawLines = string.split(LINEBREAK);
		for(String rawLine : rawLines) {
			System.out.println(rawLine);
			int curChar = 0;
			while(curChar < rawLine.length()) {
				if(rawLine.charAt(curChar) == ' ') { //strip whitespace from after line breaks
					curChar++;
				}
				System.out.println("lenght = " +  rawLine.length() + "   curChar:" +curChar + "    max "+ maxCharsPerLine);
				if((rawLine.length() - curChar)<= maxCharsPerLine) {
					System.out.println("it fits");
					
					lines.add(rawLine.substring(curChar));
					curChar = rawLine.length();
				} else {
					System.out.println("no fit");
					int endOfLine = rawLine.lastIndexOf(' ', maxCharsPerLine + curChar);
					if(endOfLine <= 0) {
						// no spaces so just set to end of line
						endOfLine = maxCharsPerLine + curChar;
					}
					lines.add(rawLine.substring(curChar, endOfLine));	
					curChar = endOfLine+1;
				}
			}

		}
		return lines;	
	}
	
}

