package net.electroland.norfolk.core;

import net.electroland.eio.InputChannel;

public class SensorEvent extends NorfolkEvent {

    protected InputChannel sourceInputChannel;
    
    public SensorEvent(){
        super();
    }

    public SensorEvent(InputChannel sourceInputChannel){
        this.sourceInputChannel = sourceInputChannel;
    }
}