package net.electroland.eio.filters;

import net.electroland.utils.ParameterMap;

public interface IOFilter {
    public void configure(ParameterMap params);
    public byte filter(byte b);
	public boolean filter(boolean b);
}