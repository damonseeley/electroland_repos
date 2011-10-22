package net.electroland.eio.filters;

public class BitLoPass implements IOFilter {

    @Override
    public byte filter(byte b) {
        return b;
    }

    @Override
    public boolean filter(boolean b) {
        return b;
    }
}