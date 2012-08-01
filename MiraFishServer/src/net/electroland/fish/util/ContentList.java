package net.electroland.fish.util;

import java.io.IOException;
import java.util.HashMap;
import java.util.Vector;

public class ContentList{


	public static  final long BUBLE_TIME_PAD = 100;

//	protected static HashMap<String, Content> THE_CONTACT_LIST;
	protected static Vector<Content> THE_CONTACT_LIST = null;
	
	static int cur = 0;


	public static Content next() {
		cur = (cur >= THE_CONTACT_LIST.size()) ? 0 : cur;
		return THE_CONTACT_LIST.get(cur++);
		
	}
//	public static HashMap<String,Content> getTheContactList() throws IOException {
	public static Vector<Content> getTheContactList() throws IOException {
		if(THE_CONTACT_LIST == null) {
			THE_CONTACT_LIST = new Vector<Content>();

//			THE_CONTACT_LIST = new HashMap<String,Content>();
			//new Reader("../MiraPool/ContentList.csv");
			new Reader("ContentList.csv");
		}
		System.out.println("Got Content List");
		return THE_CONTACT_LIST;
	}

	public static class Reader extends CSVReader {
		public Reader(String fileName) throws IOException {
			super(fileName);
		}


		public void parseLine(String[] line) {
			Content c = new Content();
			c.id = Integer.parseInt(line[0]);
			c.name = line[1];
			c.width = Integer.parseInt(line[2]);
			c.height = Integer.parseInt(line[3]);
			c.duration =    (long) (Float.parseFloat(line[4]) * 1000.0) + BUBLE_TIME_PAD;		
			c.updateHW();
			System.out.println("c  " + c + "  " + c.width + "x" + c.height);

//			THE_CONTACT_LIST.put(c.name, c);
			THE_CONTACT_LIST.add(c);
		}
	}

}
