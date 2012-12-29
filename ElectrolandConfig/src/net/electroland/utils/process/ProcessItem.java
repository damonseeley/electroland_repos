package net.electroland.utils.process;

public interface ProcessItem {

    public int getPID();

    public String getName();

    public boolean equals(ProcessItem another);
}
