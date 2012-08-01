package net.electroland.fish.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public abstract class  CSVReader {
	public CSVReader(String s) throws IOException {
		BufferedReader bfr = new BufferedReader(new FileReader(s));
		String l = bfr.readLine(); // throw away header
		
		l = bfr.readLine();
		while(l != null) {
			parseLine(l.split(","));
			l = bfr.readLine();
		}
	}
	

	public abstract void parseLine(String[] line);
	

}
