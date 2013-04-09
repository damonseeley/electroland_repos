package net.electroland.eio.filters;

import net.electroland.eio.Value;
import net.electroland.utils.ParameterMap;

abstract public class Filter {

    private String id;

    public String getId() {
        return id;
    }
    public void setId(String id) {
        this.id = id;
    }
    abstract public void configure(ParameterMap map);
    abstract public void filter(Value in);
}