package net.electroland.scSoundControl;

import java.awt.TextArea;
import java.io.BufferedInputStream;
import java.io.InputStream;

public class StreamedTextArea extends TextArea implements Runnable {

	BufferedInputStream _stream;
	byte[] _streamData;
	Thread _t;
	
	public StreamedTextArea(InputStream str) {
		_streamData = new byte[1024]; //1K buffer of text copied at a time
		
		_t = new Thread(this);
		_t.setPriority(Thread.MIN_PRIORITY);
		_t.start(); //start the monitoring thread
	}
	
	public synchronized void setInputStream(InputStream str) {
		if (str == null) _stream = null;
		else _stream = new BufferedInputStream(str);
	}
	
	//copy text from the input stream to the text area.
	//new text polling occurs every 5 milliseconds.
	public void run() {
		while(true) {
		try {
			while (_stream != null && _stream.available() > 0) {
				_stream.read(_streamData);
				this.append(new String(_streamData));
			}
		} catch (Exception e) {}
		
		try {
			Thread.yield();
		} catch (Exception e) {}
		}
	}
}
