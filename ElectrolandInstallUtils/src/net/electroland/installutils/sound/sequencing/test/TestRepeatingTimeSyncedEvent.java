package net.electroland.installutils.sound.sequencing.test;

import java.util.Map;

import net.electroland.installutils.sound.sequencing.RepeatingTimeSyncedEvent;

public class TestRepeatingTimeSyncedEvent extends RepeatingTimeSyncedEvent {

    String name;

    public TestRepeatingTimeSyncedEvent(String name, int repeats)
    {
        this.name = name;
        this.setRepeats(repeats);
    }

    @Override
    public void doEvent(Map<String, Object> context) {
        System.out.println("TestRepeatingTimeSyncedEvent." + name + "." + repeats);
    }
}