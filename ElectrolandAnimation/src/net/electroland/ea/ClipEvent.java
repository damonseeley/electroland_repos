package net.electroland.ea;

import java.util.EventObject;

public class ClipEvent extends EventObject {

    
    public int clipId;
    public Object clip;
    public int type;

    public static final int ENDED    = 0;
    public static final int STARTED  = 1;

    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    public ClipEvent(Object arg0) {
        super(arg0);
    }
}