package net.electroland.eio.devices;

import net.electroland.utils.ParameterMap;

abstract public class OutputChannel extends Channel {

    public OutputChannel(ParameterMap p) {
        super(p);
    }

    abstract public byte[] write(Object value);
}