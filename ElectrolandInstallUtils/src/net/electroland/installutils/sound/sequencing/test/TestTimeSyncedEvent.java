package net.electroland.installutils.sound.sequencing.test;

import java.util.Map;

import net.electroland.installutils.sound.sequencing.TimeSyncedEvent;

public class TestTimeSyncedEvent implements TimeSyncedEvent {

    String name;

    public TestTimeSyncedEvent(String name)
    {
        this.name = name;
    }
    @Override
    public void doEvent(Map<String, Object> context) {
        System.out.println("TestTimeSyncedEvent." + name);
    }
}
