package net.electroland.connection.animation;

public interface Animation {
	public void start();
	public void stop();
	public byte[] draw();
	public int getDefaultDuration();
}