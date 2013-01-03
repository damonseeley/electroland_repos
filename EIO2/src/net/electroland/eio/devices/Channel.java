package net.electroland.eio.devices;

import java.util.List;

import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

public abstract class Channel {

    public String id;

    protected List<Filter>filters;

    public Channel(ParameterMap p){}

    public boolean isSuspect(){
        return false;
    }
}