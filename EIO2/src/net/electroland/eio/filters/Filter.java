package net.electroland.eio.filters;

import net.electroland.eio.Value;
import net.electroland.utils.ParameterMap;

public interface Filter {
    public void configure(ParameterMap map);
    public void filter(Value in);
}