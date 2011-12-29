package net.electroland.installutils.sound.sequencing;

import java.util.Map;

abstract public class RepeatingTimeSyncedEvent implements TimeSyncedEvent {

    protected int repeats = 1;

    abstract public void doEvent(Map<String, Object> context);

    public void setRepeats(int repeats)
    {
        this.repeats = repeats;
    }
}
