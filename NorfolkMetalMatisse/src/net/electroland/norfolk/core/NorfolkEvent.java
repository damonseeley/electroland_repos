package net.electroland.norfolk.core;

abstract class NorfolkEvent {
    protected long eventTime;
    public NorfolkEvent(){
        eventTime = System.currentTimeMillis();
    }
}