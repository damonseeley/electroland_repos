package net.electroland.utils.sensors.filters;

public interface IOFilter {
	public byte filter(byte b);
	public boolean filter(boolean b);
}