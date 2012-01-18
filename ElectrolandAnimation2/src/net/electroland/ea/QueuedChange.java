package net.electroland.ea;

/**
 * represents the state of a queued change.
 * @author production
 *
 */
public class QueuedChange {
    final public static int DELETE_CHILDREN = -2;
    final public static int DELETE = -1;
    final public static int DELAY = 0;
    final public static int CHANGE = 1;
    protected int type;
    protected Change change;
    protected long duration = 0;
    protected long delay = 0;
    protected long startTime;
    protected long endTime;
    protected boolean started = false;
    
    public String toString()
    {
        StringBuffer sb = new StringBuffer();
        sb.append("type=");
        switch (type){
            case(DELETE_CHILDREN):
                sb.append("DELETE_CHILDREN, ");
            break;
            case(DELETE):
                sb.append("DELETE, ");
            break;
            case(DELAY):
                sb.append("DELAY, ");
            break;
            case(CHANGE):
                sb.append("CHANGE, ");
            break;
        }
        sb.append("duration=").append(duration);
        sb.append(", delay=").append(delay);
        sb.append(", change=").append(change);
        return sb.toString();
    }
}