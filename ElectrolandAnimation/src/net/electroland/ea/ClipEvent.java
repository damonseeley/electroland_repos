package net.electroland.ea;

import java.util.EventObject;

public class ClipEvent extends EventObject {

    
    int clipId;
    Object Clip;

    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    public ClipEvent(Object arg0) {
        super(arg0);
        // TODO Auto-generated constructor stub
    }

}