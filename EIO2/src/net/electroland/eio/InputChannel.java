package net.electroland.eio;

import java.util.ArrayList;
import java.util.List;

import net.electroland.eio.filters.Filter;

abstract public class InputChannel extends Channel {

    protected List<Filter>filters;
    protected Coordinate location;

    public Coordinate getLocation(){
        return location;
    }

    public void setLocation(Coordinate location){
        this.location = location;
    }

    public void addFilter(Filter filter){
        if (filters == null){
            filters = new ArrayList<Filter>();
        }
        filters.add(filter);
    }

    public void filter(Value value){
        if (filters != null){
            for (Filter f : filters){
                f.filter(value);
            }
        }
    }

    public String toString(){
        return this.getClass() + "InputChannel." + id + "[location=" + location + "]";
    }
}