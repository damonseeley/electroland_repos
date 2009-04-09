package net.electroland.noho.core;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Vector;


/**
 * TextQueue fetchs text from the webserver and makes it available to the rest of the program
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class TextQueue {
	long queryTime = 60*60*1000;
	Vector<TextBundle> texts = new Vector<TextBundle>();
	int cur = 0;
	boolean isRunning = true;
	TextFetcher fetcher;

	public TextBundle getNext() {
		if(cur >= texts.size()) cur = 0;
		return texts.get(cur++);
	}
	public void destroy() {
		synchronized(fetcher) {
			isRunning = false;
			fetcher.notifyAll();
		}
	}

	public boolean isReady() {
		return ! texts.isEmpty();
	}
	public TextQueue() {
		fetcher = new 	TextFetcher();
		fetcher.start();
	}

	//TODO: need a secure database connection
	//TODO: need to think about failure modes
	public class TextFetcher extends Thread {
		Vector<TextBundle> textBuff = new Vector<TextBundle>();
		BufferedReader reader;

		public void fetch() {
			BufferedReader in = null;
			try {
				URL url;
				//url = new URL("http://noho.electroland.net/db_comma_delimited_output.php");
				
				//added by DS for local file reading
				//File file = new File("./afi100_quotes_list_modified_with_breaks.txt");
				File file = new File(NoHoConfig.QUOTEFILE);
				url = file.toURI().toURL();

				in = new BufferedReader(new InputStreamReader(url.openStream()));
			} catch (MalformedURLException e2) {
				// TODO Auto-generated catch block
				e2.printStackTrace();
				return;
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
				return;
			}		
			int c;
			try {
				c = in.read();
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}

			while(c!=-1) {
				StringBuffer num = new StringBuffer(3);
				while(c!=',') {
					num.append((char)c);
					try {
						c = in.read();
					} catch (IOException e) {
						e.printStackTrace();
						return;
					}
				}
				try {
					c = in.read();
				} catch (IOException e) {
					e.printStackTrace();
					return;
				}
				StringBuffer txt = new StringBuffer(100);
				while(c!='<') {
					txt.append((char)c);
					try {
						c = in.read();
					} catch (IOException e) {
						e.printStackTrace();
						return;
					}
				}
				while(c!='>') {
					try {
						c = in.read();
					} catch (IOException e) {
						e.printStackTrace();
						return;
					}
				}
				try {
					c = in.read();
				} catch (IOException e) {
					e.printStackTrace();
					return;
				}
				
				TextBundle tb = new TextBundle(Integer.parseInt(num.toString()), txt.toString().toLowerCase());
				textBuff.add(tb);

			}
			if(! textBuff.isEmpty()) {
				texts = textBuff;
				textBuff = texts;
			}
			

		}
		public void run() {
			while(isRunning) {
				fetch();
				synchronized(this) {
					if(isRunning) {
						try {
							wait(queryTime);
						} catch (InterruptedException e) {
						}
					}
				}
			}

		}
	}

	public static void main(String ars[]) {
		new TextQueue();
	}
}
