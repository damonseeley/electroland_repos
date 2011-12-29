package net.electroland.installutils.sound.sequencing;

import java.util.Map;

public interface TimeSyncedEvent {
    public void doEvent(Map<String,Object> context);
}