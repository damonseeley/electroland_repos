package net.electroland.noho.core;

import java.util.Vector;

import net.electroland.noho.core.NoHoConfig;

/***
 * 
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class LineFormatter {
	public static final String LINEBREAK = "\n";


	static int maxCharsPerLine = NoHoConfig.DISPLAYWIDTH / NoHoConfig.CHARWIDTH; 

	//TODO: do we need padding? (IE do charaters look ok flush left and right?)

	//TODO: probably should add code to check for unrecognized chars
	/**
	 * @param string - string to be split into lines
	 * @return a vector of strings - split at line breaks and where needed to fit on screen
	 */
	public static Vector<String> formatString(String string) {
		Vector<String> lines = new Vector<String>();
		String[]  rawLines = string.split(LINEBREAK);
		for(String rawLine : rawLines) {
			int curChar = 0;
			while(curChar < rawLine.length()) {
				if(rawLine.charAt(curChar) == ' ') { //strip whitespace from after line breaks
					curChar++;
				}
				if((rawLine.length() - curChar)<= maxCharsPerLine) {
					
					lines.add(rawLine.substring(curChar));
					curChar = rawLine.length();
				} else {
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

