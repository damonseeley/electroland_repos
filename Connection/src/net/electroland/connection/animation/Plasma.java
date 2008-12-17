package net.electroland.connection.animation;

public class Plasma {

	public float xc, yc, calc1, calc2, s, s1, s2, s3;
	public int gridx, gridy, frameCount, element;
	public int xsize, ysize;
	public byte[] buffer;
	
	public Plasma(int _gridx, int _gridy){
		gridx = _gridx;
		gridy = _gridy;
		frameCount = 0;
		buffer = new byte[(gridx*gridy)*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		xsize = 10;
		ysize = 20;
	}
	
	public byte[] draw(){	
		xc = 25;
		frameCount += 1;
		calc1 = (float)Math.sin(Math.toRadians(frameCount* 0.61655617));
		calc2 = (float)Math.sin(Math.toRadians(frameCount* -3.6352262));
		
		for(int x=0; x<gridx; x++, xc+=xsize){
			yc = 25;
			s1 = (float) (128 + 128 * Math.sin(Math.toRadians(xc) * calc1));
			for(int y=0; y<gridy; y++, yc+=ysize){
				s2 = (float)(128 + 128 * Math.sin(Math.toRadians(yc) * calc2));
				s3 = (float)(128 + 128 * Math.sin(Math.toRadians((xc + yc + frameCount * 10)/2)));
				s = (s1 + s2 + s3) / 3;
				element = x+y*gridx;
				buffer[element*2 + 2] = (byte)s;
				buffer[element*2 + 3] = (byte)s2;
			}
		}
		return buffer;
	}
	
}
