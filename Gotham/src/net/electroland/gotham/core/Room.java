package net.electroland.gotham.core;

import java.util.Collection;

import net.electroland.elvis.net.GridData;

public class Room {

    private int width;
    private int height;
    private byte[] data;

    public Room(GridData d){
        width  = d.width;
        height = d.height;
        System.arraycopy(d.data, 0, data, 0, d.data.length);
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public byte[] getData() {
        return data;
    }

    public Collection<Person> getPersons() {
        return null;
    }
}