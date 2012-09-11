package net.electroland.gotham.core;

import net.electroland.elvis.net.GridData;

public class People {

    private int width;
    private int height;
    private byte[] data;

    public People(GridData d){
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
}