package net.electroland.eio.model;

import java.util.EventObject;

public class ModelEvent extends EventObject {

    public String watcherName;
    /**
     * 
     */
    private static final long serialVersionUID = -6351993345221172615L;

    public ModelEvent(Object arg0) {
        super(arg0);
    }
}