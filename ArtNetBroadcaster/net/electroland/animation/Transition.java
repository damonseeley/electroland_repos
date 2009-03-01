package net.electroland.animation;

public class Transition implements Animation {

	private Animation begin, end;
	
	public Transition(Animation begin, Animation end){
		this.begin = begin;
		this.end = end;
	}

	public void cleanUp() {
	}

	public Raster getFrame() {
		// composite begin and end
		return null;
	}

	public void initialize() {		
	}

	public boolean isDone() {
		// TODO Auto-generated method stub
		return false;
	}
}