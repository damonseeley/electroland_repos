package net.electroland.norfolk.eio.filters;

public class PersonEvent {

    public enum Direction {LEFT, RIGHT, CENTER};
    public enum Behavior {ENTER, EXIT}; // this might also be unknowable

    private Direction direction;
    private Behavior behavior;
    private String channelId;

    public PersonEvent(String channelId, Direction direction, Behavior behavior){
        this.channelId = channelId;
        this.direction = direction;
        this.behavior  = behavior;
    }

    public Behavior getBehavior() {
        return behavior;
    }

    public Direction getDirection() {
        return direction;
    }

    public String getChannelId() {
        return channelId;
    }
}