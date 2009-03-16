package net.electroland.util;

import java.util.HashMap;
import java.util.Map;

public class OptionParser {

	private static String ARG_MARKER = " -";

	/**
	 * This is a primitive parse for [-key val] style option variables in Strings.
	 * For example, for the input:
	 * 
	 * 	-key1 val1 -key2 val2 val3 -key3 val4
	 * 
	 * We'll return a Map with the following key value pairs:
	 * 
	 * 	key		value
	 * 	------	---------
	 * 	-key1	val1
	 * 	-key2	val2 val3
	 * 	-key3	val4
	 * 
	 * WARNING: This parser knowingly converts all tabs to spaces, so any 
	 * 			arguments you pass it that contain tabs will be affected as such.
	 * 
	 * @param str
	 * @return a Map of the keys and their values.
	 * @throws OptionException if the string does not properly start with a flag. 
	 */
	public static Map<String, Object> parse(String str) throws OptionException
	{
		HashMap <String, Object> map = new HashMap<String, Object>();
		if (str == null)
		{
			return map;
		}

		// all we are really doing is tokenizing on " -" (ARG_MARKER).  Then
		// splitting each token on the first space into a kehy and a value.

		// sorry.  no tabs.
		str.replace('\t', ' ');
		// make sure the first and last tokens are delimited.
		str = ' ' + str.trim() + ARG_MARKER; 

		int flagStart = str.indexOf(ARG_MARKER);

		// SPECIAL CASE: first token (or only token) doesn't start with "-".
		if (!str.startsWith(ARG_MARKER))
		{
			throw new OptionException("Unknown option " + 
										str.substring(1, flagStart));
		}

		int realEnd = str.length() - 2;
		while (true)
		{
			if (flagStart == realEnd) // out of tokens.
			{
				break;
			} else {
				int nextFlagStart = str.indexOf(ARG_MARKER, flagStart + 2);
				String tok = str.substring(flagStart, nextFlagStart).trim();
				int flagEnd = tok.indexOf(' ');
				if (flagEnd == -1){
					map.put(tok, null); // SPECIAL CASE: flag has no value.
				}else{
					map.put(tok.substring(0, flagEnd), 
							tok.substring(flagEnd + 1, tok.length()));
					// at some point, we should parse the values attributed to
					// each flag into an object here.
				}
				flagStart = nextFlagStart;
			}
		}
		return map;
	}
}