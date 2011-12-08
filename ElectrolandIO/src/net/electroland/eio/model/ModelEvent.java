package net.electroland.eio.model;

import java.util.EventObject;
import java.util.Map;

public class ModelEvent extends EventObject {

    public String watcherName;
    public Map<String, Object> optionalPostiveDetails;
    /**
     * 
     */
    private static final long serialVersionUID = -6351993345221172615L;

    public ModelEvent(Object arg0) {
        super(arg0);
    }
}