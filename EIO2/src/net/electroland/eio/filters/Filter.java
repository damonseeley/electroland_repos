package net.electroland.eio.filters;

import net.electroland.utils.ParameterMap;

public interface Filter {
    public void configure(ParameterMap map);
    public Object filter(Object in);
}