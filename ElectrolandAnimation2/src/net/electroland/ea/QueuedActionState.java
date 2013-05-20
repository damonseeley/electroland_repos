package net.electroland.ea;

/**
 * represents the state of a queued tween/message/delay.
 * 
 * Name is a little inacurate, since sometimes we're not tweening- we're a
 * delay or a message.
 * 
 * @author production
 *
 */
public class QueuedActionState {
    
    enum Type {DELETE_CHILDREN, DELETE, DELAY, CHANGE, MESSAGE}

    protected Type    type;
    protected Tween   change;
    protected long    duration     = 0;
    protected long    delay        = 0;
    protected long    startTime;
    protected long    endTime;
    protected boolean started   = false;
}