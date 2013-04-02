package net.electroland.eio.devices.modbus;

import net.electroland.eio.InputChannel;

public class ModBusTcpInputChannel extends InputChannel {

    public enum Type {
        BYTE, SHORT, UNINT
    }

    protected Type type;

    public String toString(){
        return super.toString() + " of type " + type;
    }
}