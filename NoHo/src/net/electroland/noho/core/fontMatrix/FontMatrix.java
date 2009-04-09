package net.electroland.noho.core.fontMatrix;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Enumeration;
import java.util.Hashtable;

public class FontMatrix {

	private Hashtable<Character, LetterformImage> letterforms = new Hashtable<Character, LetterformImage>();

	private Hashtable<Character, String> charNameMap = new Hashtable<Character, String>();

	public LetterformImage testFont;

	String fontImageDir;

	public FontMatrix(String dirName) {
		fontImageDir = dirName;

		//fill a hashtable of character/name pairs for fetching image files
		fillCharNameMap();

		// create letterformimage objects for each character we need to map
		// each LFI contains both a buffered image and bytematrix version of the letterform
		for (Enumeration e = charNameMap.keys(); e.hasMoreElements();) {
			char theChar = e.nextElement().toString().charAt(0);
			String theValue = charNameMap.get(theChar);
			//System.out.println(theChar + "  " + theValue);

			//construct full path filenames
			String letterformImageFilePath = fontImageDir + theValue + ".gif";
			try {
				//create a temp LFI and add it to the letterforms hashtable
				LetterformImage tempLFI = new LetterformImage(letterformImageFilePath);
				letterforms.put(theChar, tempLFI);
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				System.out.println("ERROR ON READ: " + theValue + ".gif");
				e1.printStackTrace();
			}

		}

		System.out.println("letterforms matrix count: " + letterforms.size());

	}

	// returns the buffered image for any char that is in the hashtable
	// currently returns a space if no char/image mapping can be found
	public BufferedImage getLetterformImg(char theChar) throws RuntimeException {
		try {
			return letterforms.get(theChar).img;
		} catch (RuntimeException e) {
			// TODO Auto-generated catch block
			System.out.println("failed getting img for char " + theChar);
			e.printStackTrace();
		}
		return letterforms.get(' ').img;
	}

	private void fillCharNameMap() {

		// add chars a to z
		for (char c = 'a'; c <= 'z'; c++) {
			String theChar = "" + c;
			charNameMap.put(c, theChar);
		}

		// add digits 0 to 9
		for (char c = '0'; c <= '9'; c++) {
			String theChar = "" + c;
			charNameMap.put(c, theChar);
		}

		// add special characters space !@#$%^&*(),.<>/?;:’”
		charNameMap.put(' ', "space");
		charNameMap.put('!', "exclamation");
		charNameMap.put('@', "at");
		charNameMap.put('#', "pound");
		charNameMap.put('$', "dollarsign");
		charNameMap.put('%', "percent");
		charNameMap.put('^', "caret");
		charNameMap.put('&', "ampersand");
		charNameMap.put('*', "asterisk");
		charNameMap.put('(', "parenleft");
		charNameMap.put(')', "parenright");
		charNameMap.put(',', "comma");
		charNameMap.put('.', "period");
		charNameMap.put('\'', "apostrophe");
		charNameMap.put('\"', "quote");
		charNameMap.put(':', "colon");
		charNameMap.put(';', "semicolon");
		charNameMap.put('?', "questionmark");
		charNameMap.put('<', "lessthan");
		charNameMap.put('>', "greaterthan");
		charNameMap.put('/', "slash");
		charNameMap.put('-', "minus");
		charNameMap.put('+', "plus");
	}

}
