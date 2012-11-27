package net.electroland.eio.devices;

import net.electroland.utils.ParameterMap;

abstract public class InputChannel extends Channel {

    public InputChannel(ParameterMap p) {
        super(p);
    }

    abstract public Object read(byte[] b);
}