package net.electroland.elvis.net;

import java.nio.ByteBuffer;
import java.util.StringTokenizer;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public  class GridData implements StringAppender {
	public int width;
	public int height;
	public byte[] data;

	public GridData(String s) {
		StringTokenizer tokenizer = new StringTokenizer(s,",");
		if(! tokenizer.hasMoreTokens()) {
			width =0;
			height =0;
			data = new byte[0];
		} else {
			try {
				width = Integer.parseInt(tokenizer.nextToken());
				height = Integer.parseInt(tokenizer.nextToken());
				data = new byte[width * height];
				for(int i=0; i < data.length; i++) {
					data[i] =  Byte.parseByte(tokenizer.nextToken());
				}
			} catch(RuntimeException e) {
				e.printStackTrace();

			}
		}

	}

	public GridData(IplImage img) {
		width = img.width();
		height = img.height();
		data = new byte[width*height];
		ByteBuffer bb = img.getByteBuffer();
		bb.get(data);
	}
	@Override
	public void buildString(StringBuilder sb) {
		sb.append(width);
		sb.append(",");
		sb.append(height);
		for(byte b : data) {
			sb.append(",");
			sb.append(b);
		}			
	}

}