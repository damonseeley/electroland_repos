package net.electroland.lighting.detector.animation;

public interface Completable {
	public boolean isDone();
	abstract public void initialize();
	abstract public void cleanUp();
	
}